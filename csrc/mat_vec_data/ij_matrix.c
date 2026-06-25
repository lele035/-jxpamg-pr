//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  ij_matrix.c -- basic operations for IJ matrices.
 *  Date: 2015/11/28
 */ 

#include "jx_mv.h"

JX_Int
JX_IJMatrixCreate( MPI_Comm     comm,
                   JX_Int          ilower,
                   JX_Int          iupper,
                   JX_Int          jlower,
                   JX_Int          jupper,
                   JX_IJMatrix *matrix )
{
   JX_Int *row_partitioning;
   JX_Int *col_partitioning;
   JX_Int *info;
   JX_Int num_procs;
   JX_Int myid;

   jx_IJMatrix *ijmatrix;

#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int  row0, col0, rowN, colN;
#else
   JX_Int *recv_buf;
   JX_Int i, i4;
   JX_Int square;
#endif

   ijmatrix = jx_CTAlloc(jx_IJMatrix, 1);

   jx_IJMatrixComm(ijmatrix)         = comm;
   jx_IJMatrixObject(ijmatrix)       = NULL;
   jx_IJMatrixTranslator(ijmatrix)   = NULL;
   jx_IJMatrixAssumedPart(ijmatrix)  = NULL;
   jx_IJMatrixObjectType(ijmatrix)   = JX_UNITIALIZED;
   jx_IJMatrixAssembleFlag(ijmatrix) = 0;
   jx_IJMatrixPrintLevel(ijmatrix)   = 0;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &myid);

   if (ilower > iupper+1 || ilower < 0)
   {
      jx_error_in_arg(2);
      jx_TFree(ijmatrix);
      return jx_error_flag;
   }

   if (iupper < -1)
   {
      jx_error_in_arg(3);
      jx_TFree(ijmatrix);
      return jx_error_flag;
   }

   if (jlower > jupper+1 || jlower < 0)
   {
      jx_error_in_arg(4);
      jx_TFree(ijmatrix);
      return jx_error_flag;
   }

   if (jupper < -1)
   {
      jx_error_in_arg(5);
      jx_TFree(ijmatrix);
      return jx_error_flag;
   }

#ifdef JX_NO_GLOBAL_PARTITION

   info = jx_CTAlloc(JX_Int, 2);

   row_partitioning = jx_CTAlloc(JX_Int, 2);
   col_partitioning = jx_CTAlloc(JX_Int, 2);

   row_partitioning[0] = ilower;
   row_partitioning[1] = iupper + 1;
   col_partitioning[0] = jlower;
   col_partitioning[1] = jupper + 1;

   /* now we need the global number of rows and columns as well
      as the global first row and column index */

   /* proc 0 has the first row and col */
   if (myid == 0)
   {
      info[0] = ilower;
      info[1] = jlower;
   }
   jx_MPI_Bcast(info, 2, JX_MPI_INT, 0, comm);
   row0 = info[0];
   col0 = info[1];

   /* proc (num_procs-1) has the last row and col */
   if (myid == (num_procs-1))
   {
      info[0] = iupper;
      info[1] = jupper;
   }
   jx_MPI_Bcast(info, 2, JX_MPI_INT, num_procs-1, comm);

   rowN = info[0];
   colN = info[1];

   jx_IJMatrixGlobalFirstRow(ijmatrix) = row0;
   jx_IJMatrixGlobalFirstCol(ijmatrix) = col0;
   jx_IJMatrixGlobalNumRows(ijmatrix) = rowN - row0 + 1;
   jx_IJMatrixGlobalNumCols(ijmatrix) = colN - col0 + 1;

   jx_TFree(info);

#else

   info = jx_CTAlloc(JX_Int, 4);
   recv_buf = jx_CTAlloc(JX_Int, 4*num_procs);
   row_partitioning = jx_CTAlloc(JX_Int, num_procs+1);

   info[0] = ilower;
   info[1] = iupper;
   info[2] = jlower;
   info[3] = jupper;

   /* Generate row- and column-partitioning through information exchange
      across all processors, check whether the matrix is square, and
      if the partitionings match. i.e. no overlaps or gaps,
      if there are overlaps or gaps in the row partitioning or column
      partitioning , ierr will be set to -9 or -10, respectively */

   jx_MPI_Allgather(info, 4, JX_MPI_INT, recv_buf, 4, JX_MPI_INT, comm);

   row_partitioning[0] = recv_buf[0];
   square = 1;
   for (i=0; i < num_procs-1; i++)
   {
      i4 = 4*i;
      if ( recv_buf[i4+1] != (recv_buf[i4+4]-1) )
      {
         jx_error(JX_ERROR_GENERIC);
         jx_TFree(ijmatrix);
         jx_TFree(info);
         jx_TFree(recv_buf);
         jx_TFree(row_partitioning);
   	 return jx_error_flag;
      }
      else
      {
         row_partitioning[i+1] = recv_buf[i4+4];
      }

      if ((square && (recv_buf[i4] != recv_buf[i4+2])) || (recv_buf[i4+1] != recv_buf[i4+3]))
      {
         square = 0;
      }
   }
   i4 = (num_procs-1)*4;
   row_partitioning[num_procs] = recv_buf[i4+1]+1;

   if ((recv_buf[i4] != recv_buf[i4+2]) || (recv_buf[i4+1] != recv_buf[i4+3]))
   {
      square = 0;
   }

   if (square) col_partitioning = row_partitioning;
   else
   {
      col_partitioning = jx_CTAlloc(JX_Int, num_procs+1);
      col_partitioning[0] = recv_buf[2];
      for (i=0; i < num_procs-1; i++)
      {
         i4 = 4*i;
         if (recv_buf[i4+3] != recv_buf[i4+6]-1)
         {
           jx_error(JX_ERROR_GENERIC);
           jx_TFree(ijmatrix);
           jx_TFree(info);
           jx_TFree(recv_buf);
           jx_TFree(row_partitioning);
           jx_TFree(col_partitioning);
   	   return jx_error_flag;
         }
         else
   	   col_partitioning[i+1] = recv_buf[i4+6];
      }
      col_partitioning[num_procs] = recv_buf[num_procs*4-1]+1;
   }

   jx_IJMatrixGlobalFirstRow(ijmatrix) = row_partitioning[0];
   jx_IJMatrixGlobalFirstCol(ijmatrix) = col_partitioning[0];
   jx_IJMatrixGlobalNumRows(ijmatrix) = row_partitioning[num_procs] - row_partitioning[0];
   jx_IJMatrixGlobalNumCols(ijmatrix) = col_partitioning[num_procs] - col_partitioning[0];

   jx_TFree(info);
   jx_TFree(recv_buf);

#endif

   jx_IJMatrixRowPartitioning(ijmatrix) = row_partitioning;
   jx_IJMatrixColPartitioning(ijmatrix) = col_partitioning;

  *matrix = (JX_IJMatrix) ijmatrix;

   return jx_error_flag;
}

JX_Int
jx_IJMatrixCreateParCSR( jx_IJMatrix *matrix )
{
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   JX_Int *row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   JX_Int *col_partitioning = jx_IJMatrixColPartitioning(matrix);
   jx_ParCSRMatrix *par_matrix;
   JX_Int *row_starts;
   JX_Int *col_starts;
   JX_Int num_procs;
   JX_Int i;

   jx_MPI_Comm_size(comm, &num_procs);

#ifdef JX_NO_GLOBAL_PARTITION
   row_starts = jx_CTAlloc(JX_Int, 2);
   if (jx_IJMatrixGlobalFirstRow(matrix))
      for (i=0; i < 2; i++)
	 row_starts[i] = row_partitioning[i] - jx_IJMatrixGlobalFirstRow(matrix);
   else
      for (i=0; i < 2; i++)
	 row_starts[i] = row_partitioning[i];
   if (row_partitioning != col_partitioning)
   {
      col_starts = jx_CTAlloc(JX_Int, 2);
      if (jx_IJMatrixGlobalFirstCol(matrix))
	 for (i=0; i < 2; i++)
	    col_starts[i] = col_partitioning[i] - jx_IJMatrixGlobalFirstCol(matrix);
      else
	 for (i=0; i < 2; i++)
	    col_starts[i] = col_partitioning[i];
   }
   else
      col_starts = row_starts;

   par_matrix = jx_ParCSRMatrixCreate(comm, jx_IJMatrixGlobalNumRows(matrix),
                                         jx_IJMatrixGlobalNumCols(matrix),
                                         row_starts, col_starts, 0, 0, 0);

#else
   row_starts = jx_CTAlloc(JX_Int, num_procs+1);
   if (row_partitioning[0])
      for (i=0; i < num_procs+1; i++)
	 row_starts[i] = row_partitioning[i] - row_partitioning[0];
   else
      for (i=0; i < num_procs+1; i++)
	 row_starts[i] = row_partitioning[i];
   if (row_partitioning != col_partitioning)
   {
      col_starts = jx_CTAlloc(JX_Int, num_procs+1);
      if (col_partitioning[0])
	 for (i=0; i < num_procs+1; i++)
	    col_starts[i] = col_partitioning[i] - col_partitioning[0];
      else
	 for (i=0; i < num_procs+1; i++)
	    col_starts[i] = col_partitioning[i];
   }
   else
      col_starts = row_starts;
   par_matrix = jx_ParCSRMatrixCreate(comm, row_starts[num_procs],
                                         col_starts[num_procs],
                                         row_starts, col_starts, 0, 0, 0);
#endif

   jx_IJMatrixObject(matrix) = par_matrix;

   return jx_error_flag;
}

JX_Int
jx_AuxParCSRMatrixCreate( jx_AuxParCSRMatrix **aux_matrix,
			  JX_Int  local_num_rows,
                       	  JX_Int  local_num_cols,
			  JX_Int *sizes )
{
   jx_AuxParCSRMatrix  *matrix;

   matrix = jx_CTAlloc(jx_AuxParCSRMatrix, 1);

   jx_AuxParCSRMatrixLocalNumRows(matrix) = local_num_rows;
   jx_AuxParCSRMatrixLocalNumCols(matrix) = local_num_cols;

   if (sizes)
   {
      jx_AuxParCSRMatrixRowSpace(matrix) = sizes;
   }
   else
   {
      jx_AuxParCSRMatrixRowSpace(matrix) = NULL;
   }

   /* set defaults */
   jx_AuxParCSRMatrixNeedAux(matrix) = 1;
   jx_AuxParCSRMatrixMaxOffProcElmts(matrix) = 0;
   jx_AuxParCSRMatrixCurrentNumElmts(matrix) = 0;
   jx_AuxParCSRMatrixOffProcIIndx(matrix) = 0;
   jx_AuxParCSRMatrixRowLength(matrix) = NULL;
   jx_AuxParCSRMatrixAuxJ(matrix) = NULL;
   jx_AuxParCSRMatrixAuxData(matrix) = NULL;
   jx_AuxParCSRMatrixIndxDiag(matrix) = NULL;
   jx_AuxParCSRMatrixIndxOffd(matrix) = NULL;
   /* stash for setting or adding off processor values */
   jx_AuxParCSRMatrixOffProcI(matrix) = NULL;
   jx_AuxParCSRMatrixOffProcJ(matrix) = NULL;
   jx_AuxParCSRMatrixOffProcData(matrix) = NULL;
  *aux_matrix = matrix;
   
   return 0;
}

JX_Int
JX_IJMatrixInitialize( JX_IJMatrix matrix )
{
   jx_IJMatrix *ijmatrix = (jx_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   if ( jx_IJMatrixObjectType(ijmatrix) == JX_PARCSR )
      jx_IJMatrixInitializeParCSR(ijmatrix);
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
jx_IJMatrixInitializeParCSR( jx_IJMatrix *matrix )
{
   jx_ParCSRMatrix *par_matrix = jx_IJMatrixObject(matrix);
   jx_AuxParCSRMatrix *aux_matrix = jx_IJMatrixTranslator(matrix);
   JX_Int local_num_rows;

   if (jx_IJMatrixAssembleFlag(matrix) == 0)
   {
      if (!par_matrix)
      {
         jx_IJMatrixCreateParCSR(matrix);
         par_matrix = jx_IJMatrixObject(matrix);
      }
      local_num_rows = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(par_matrix));
      if (!aux_matrix)
      {
         jx_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
            jx_CSRMatrixNumCols(jx_ParCSRMatrixDiag(par_matrix)), NULL);
         jx_IJMatrixTranslator(matrix) = aux_matrix;
      }

      jx_ParCSRMatrixInitialize(par_matrix);
      jx_AuxParCSRMatrixInitialize(aux_matrix);
      if (!jx_AuxParCSRMatrixNeedAux(aux_matrix))
      {
         JX_Int i, *indx_diag, *indx_offd, *diag_i, *offd_i;
         diag_i = jx_CSRMatrixI(jx_ParCSRMatrixDiag(par_matrix));
         offd_i = jx_CSRMatrixI(jx_ParCSRMatrixOffd(par_matrix));
         indx_diag = jx_AuxParCSRMatrixIndxDiag(aux_matrix);
         indx_offd = jx_AuxParCSRMatrixIndxOffd(aux_matrix);
#if JX_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
         for (i=0; i < local_num_rows; i++)
         {
	    indx_diag[i] = diag_i[i];
	    indx_offd[i] = offd_i[i];
         }
      }
   }
   else
   {
      if (!aux_matrix)
      {
         local_num_rows = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(par_matrix));
         jx_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows,
            jx_CSRMatrixNumCols(jx_ParCSRMatrixDiag(par_matrix)), NULL);
         jx_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
         jx_IJMatrixTranslator(matrix) = aux_matrix;
      }
   }

   return jx_error_flag;
}

JX_Int
jx_AuxParCSRMatrixInitialize( jx_AuxParCSRMatrix *matrix )
{
   JX_Int local_num_rows = jx_AuxParCSRMatrixLocalNumRows(matrix);
   JX_Int *row_space = jx_AuxParCSRMatrixRowSpace(matrix);
   JX_Int max_off_proc_elmts = jx_AuxParCSRMatrixMaxOffProcElmts(matrix);
   JX_Int **aux_j;
   JX_Real **aux_data;
   JX_Int i;

   if (local_num_rows < 0)
      return -1;
   if (local_num_rows == 0)
      return 0;
   /* allocate stash for setting or adding off processor values */
   if (max_off_proc_elmts > 0)
   {
      jx_AuxParCSRMatrixOffProcI(matrix) = jx_CTAlloc(JX_Int, 2*max_off_proc_elmts);
      jx_AuxParCSRMatrixOffProcJ(matrix) = jx_CTAlloc(JX_Int, max_off_proc_elmts);
      jx_AuxParCSRMatrixOffProcData(matrix) = jx_CTAlloc(JX_Real, max_off_proc_elmts);
   }
   if (jx_AuxParCSRMatrixNeedAux(matrix))
   {
      aux_j = jx_CTAlloc(JX_Int *, local_num_rows);
      aux_data = jx_CTAlloc(JX_Real *, local_num_rows);
      if (!jx_AuxParCSRMatrixRowLength(matrix))
         jx_AuxParCSRMatrixRowLength(matrix) = jx_CTAlloc(JX_Int, local_num_rows);
      if (row_space)
      {
         for (i=0; i < local_num_rows; i++)
         {
            aux_j[i] = jx_CTAlloc(JX_Int, row_space[i]);
            aux_data[i] = jx_CTAlloc(JX_Real, row_space[i]);
         }
      }
      else
      {
         row_space = jx_CTAlloc(JX_Int, local_num_rows);
         for (i=0; i < local_num_rows; i++)
         {
            row_space[i] = 30;
            aux_j[i] = jx_CTAlloc(JX_Int, 30);
            aux_data[i] = jx_CTAlloc(JX_Real, 30);
         }
         jx_AuxParCSRMatrixRowSpace(matrix) = row_space;
      }
      jx_AuxParCSRMatrixAuxJ(matrix) = aux_j;
      jx_AuxParCSRMatrixAuxData(matrix) = aux_data;
   }
   else
   {
      jx_AuxParCSRMatrixIndxDiag(matrix) = jx_CTAlloc(JX_Int, local_num_rows);
      jx_AuxParCSRMatrixIndxOffd(matrix) = jx_CTAlloc(JX_Int, local_num_rows);
   }

   return 0;
}

JX_Int
JX_IJMatrixDestroy( JX_IJMatrix matrix )
{
   jx_IJMatrix *ijmatrix = (jx_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   if (ijmatrix)
   {
      if (jx_IJMatrixRowPartitioning(ijmatrix) == jx_IJMatrixColPartitioning(ijmatrix))
         jx_TFree(jx_IJMatrixRowPartitioning(ijmatrix));
      else
      {
         jx_TFree(jx_IJMatrixRowPartitioning(ijmatrix));
         jx_TFree(jx_IJMatrixColPartitioning(ijmatrix));
      }
      if (jx_IJMatrixAssumedPart(ijmatrix))
         jx_AssumedPartitionDestroy(jx_IJMatrixAssumedPart(ijmatrix));
      if (jx_IJMatrixObjectType(ijmatrix) == JX_PARCSR)
         jx_IJMatrixDestroyParCSR(ijmatrix);
      else if (jx_IJMatrixObjectType(ijmatrix) != -1)
      {
         jx_error_in_arg(1);
         return jx_error_flag;
      }
   }

   jx_TFree(ijmatrix);

   return jx_error_flag;
}

JX_Int
jx_IJMatrixDestroyParCSR( jx_IJMatrix *matrix )
{
   jx_ParCSRMatrixDestroy(jx_IJMatrixObject(matrix));
   jx_AuxParCSRMatrixDestroy(jx_IJMatrixTranslator(matrix));
   return jx_error_flag;
}

JX_Int
jx_AuxParCSRMatrixDestroy( jx_AuxParCSRMatrix *matrix )
{
   JX_Int ierr=0;
   JX_Int i;
   JX_Int num_rows;

   if (matrix)
   {
      num_rows = jx_AuxParCSRMatrixLocalNumRows(matrix);
      if (jx_AuxParCSRMatrixRowLength(matrix))
         jx_TFree(jx_AuxParCSRMatrixRowLength(matrix));
      if (jx_AuxParCSRMatrixRowSpace(matrix))
         jx_TFree(jx_AuxParCSRMatrixRowSpace(matrix));
      if (jx_AuxParCSRMatrixAuxJ(matrix))
      {
         for (i=0; i < num_rows; i++)
	    jx_TFree(jx_AuxParCSRMatrixAuxJ(matrix)[i]);
	 jx_TFree(jx_AuxParCSRMatrixAuxJ(matrix));
      }
      if (jx_AuxParCSRMatrixAuxData(matrix))
      {
         for (i=0; i < num_rows; i++)
            jx_TFree(jx_AuxParCSRMatrixAuxData(matrix)[i]);
	 jx_TFree(jx_AuxParCSRMatrixAuxData(matrix));
      }
      if (jx_AuxParCSRMatrixIndxDiag(matrix))
            jx_TFree(jx_AuxParCSRMatrixIndxDiag(matrix));
      if (jx_AuxParCSRMatrixIndxOffd(matrix))
            jx_TFree(jx_AuxParCSRMatrixIndxOffd(matrix));
      if (jx_AuxParCSRMatrixOffProcI(matrix))
      	    jx_TFree(jx_AuxParCSRMatrixOffProcI(matrix));
      if (jx_AuxParCSRMatrixOffProcJ(matrix))
      	    jx_TFree(jx_AuxParCSRMatrixOffProcJ(matrix));
      if (jx_AuxParCSRMatrixOffProcData(matrix))
      	    jx_TFree(jx_AuxParCSRMatrixOffProcData(matrix));
      jx_TFree(matrix);
   }

   return ierr;
}

JX_Int
jx_AssumedPartitionDestroy( jx_IJAssumedPart *apart )
{
   if (apart->storage_length > 0)
   {
      jx_TFree(apart->proc_list);
      jx_TFree(apart->row_start_list);
      jx_TFree(apart->row_end_list);
      jx_TFree(apart->sort_index);
   }
   jx_TFree(apart);

   return jx_error_flag;
}

JX_Int
JX_IJMatrixSetObjectType( JX_IJMatrix matrix, JX_Int type )
{
   jx_IJMatrix *ijmatrix = (jx_IJMatrix *) matrix;
   if (!ijmatrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_IJMatrixObjectType(ijmatrix) = type;

   return jx_error_flag;
}

JX_Int
JX_IJMatrixSetValues( JX_IJMatrix    matrix,
                      JX_Int            nrows,
                      JX_Int           *ncols,
                      const JX_Int     *rows,
                      const JX_Int     *cols,
                      const JX_Real  *values )
{
   jx_IJMatrix *ijmatrix = (jx_IJMatrix *) matrix;

   if (nrows == 0)
      return jx_error_flag;

   if (!ijmatrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   if (nrows < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (!ncols)
   {
      jx_error_in_arg(3);
      return jx_error_flag;
   }

   if (!rows)
   {
      jx_error_in_arg(4);
      return jx_error_flag;
   }

   if (!cols)
   {
      jx_error_in_arg(5);
      return jx_error_flag;
   }

   if (!values)
   {
      jx_error_in_arg(6);
      return jx_error_flag;
   }

   if (jx_IJMatrixObjectType(ijmatrix) == JX_PARCSR)
   {
      if (jx_IJMatrixOMPFlag(ijmatrix))
	 return(jx_IJMatrixSetValuesOMPParCSR(ijmatrix, nrows, ncols, rows, cols, values));
      else
         return(jx_IJMatrixSetValuesParCSR(ijmatrix, nrows, ncols, rows, cols, values));
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
jx_IJMatrixSetValuesOMPParCSR( jx_IJMatrix  *matrix,
                               JX_Int           nrows,
                               JX_Int          *ncols,
                               const JX_Int    *rows,
                               const JX_Int    *cols,
                               const JX_Real *values )
{
   jx_ParCSRMatrix *par_matrix;
   jx_CSRMatrix *diag, *offd;
   jx_AuxParCSRMatrix *aux_matrix;
   JX_Int *row_partitioning;
   JX_Int *col_partitioning;
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   JX_Int num_procs, my_id;
   JX_Int col_0, col_n;
   JX_Int cancel_indx = 0;
   JX_Int **aux_j = NULL;
   JX_Real **aux_data = NULL;
   JX_Int *row_length, *row_space;
   JX_Int need_aux;
   JX_Int *diag_i = NULL;
   JX_Int *diag_j = NULL;
   JX_Real *diag_data = NULL;
   JX_Int *offd_i = NULL;
   JX_Int *offd_j = NULL;
   JX_Real *offd_data = NULL;
   JX_Int first, pstart;
   //JX_Int current_num_elmts;
   /*JX_Int max_off_proc_elmts;*/
   JX_Int off_proc_i_indx = 0;
   JX_Int *off_proc_i = NULL;
   JX_Int *off_proc_j = NULL;
   JX_Int *value_start, *offproc_cnt;

   JX_Int print_level = jx_IJMatrixPrintLevel(matrix);
   JX_Int max_num_threads;
   JX_Int error_flag = 0;
   JX_Int i1;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   max_num_threads = jx_NumThreads();
   par_matrix = jx_IJMatrixObject(matrix);
   row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   col_partitioning = jx_IJMatrixColPartitioning(matrix);

   value_start = jx_CTAlloc(JX_Int, max_num_threads+1);
   offproc_cnt = jx_CTAlloc(JX_Int, max_num_threads);

#ifdef JX_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first =  jx_IJMatrixGlobalFirstCol(matrix);
   pstart = 0;
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
   pstart = my_id;
#endif
   if (nrows < 0)
   {
      jx_error_in_arg(2);
      if (print_level)
         jx_printf("Error! nrows negative! JX_IJMatrixSetValues\n");
      return jx_error_flag;
   }

   if (jx_IJMatrixAssembleFlag(matrix))  /* matrix already assembled*/
   {
      JX_Int *col_map_offd = NULL;
      JX_Int num_cols_offd;

      diag = jx_ParCSRMatrixDiag(par_matrix);
      diag_i = jx_CSRMatrixI(diag);
      diag_j = jx_CSRMatrixJ(diag);
      diag_data = jx_CSRMatrixData(diag);
      offd = jx_ParCSRMatrixOffd(par_matrix);
      offd_i = jx_CSRMatrixI(offd);
      num_cols_offd = jx_CSRMatrixNumCols(offd);
      if (num_cols_offd)
      {
          col_map_offd = jx_ParCSRMatrixColMapOffd(par_matrix);
          offd_j = jx_CSRMatrixJ(offd);
          offd_data = jx_CSRMatrixData(offd);
      }
      aux_matrix = jx_IJMatrixTranslator(matrix);
      if (aux_matrix)
      {
//         current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
         off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
         cancel_indx = jx_AuxParCSRMatrixCancelIndx(aux_matrix);
      }

#if JX_USING_OPENMP
#pragma omp parallel 
#endif
      {
         JX_Int j_offd;
         JX_Int num_threads, my_thread_num;
         JX_Int len, rest, ns, ne;
         JX_Int pos_diag, pos_offd;
         JX_Int len_diag, len_offd;
         JX_Int row_len;
         JX_Int row_local;
         JX_Int i, j, k, ii, n, row;
         JX_Int not_found, size, indx, cnt1, col_indx;

         num_threads = jx_NumActiveThreads();
         my_thread_num = jx_GetThreadNum();

         len = nrows/num_threads; 
         rest = nrows - len*num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num*(len+1);
            ne = (my_thread_num+1)*(len+1);
         }
         else
         {
            ns = my_thread_num*len+rest;
            ne = (my_thread_num+1)*len+rest;
         }

         value_start[my_thread_num] = 0;
         for (ii=ns; ii < ne; ii++)
            value_start[my_thread_num] += ncols[ii];

#if JX_USING_OPENMP
#pragma omp barrier
#endif  
         if (my_thread_num == 0)
         {
            for (i=0; i < max_num_threads; i++)
               value_start[i+1] += value_start[i];
         }
#if JX_USING_OPENMP
#pragma omp barrier
#endif  
         indx = 0;
	 if (my_thread_num)
	   indx = value_start[my_thread_num-1];
         for (ii=ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols[ii];
            /* processor owns the row */ 
            if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
            {
               row_local = row - row_partitioning[pstart];

               /* compute local row number */
               size = diag_i[row_local+1] - diag_i[row_local] + offd_i[row_local+1] - offd_i[row_local];

               if (n > size)
               {
                  jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                  error_flag++;
      	          if (print_level) jx_printf (" row %d too long! \n", row);
      	          break;
               }
       
               pos_diag = diag_i[row_local];
               pos_offd = offd_i[row_local];
               len_diag = diag_i[row_local+1];
               len_offd = offd_i[row_local+1];
               not_found = 1;
      	
               for (i=0; i < n; i++)
               {
                  if (cols[indx] < col_0 || cols[indx] > col_n)
                  /* insert into offd */	
                  {
      	             j_offd = jx_BinarySearch(col_map_offd,cols[indx]-first,
                                              num_cols_offd);
      	             if (j_offd == -1)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
      	                break;
      	             }
      	             for (j=pos_offd; j < len_offd; j++)
      	             {
      	                if (offd_j[j] == j_offd)
      	                {
                           offd_data[j] = values[indx];
      		           not_found = 0;
      		           break;
      	                }
      	             }
      	             if (not_found)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
      	                break;
      	             }
      	             not_found = 1;
                  }
                  /* diagonal element */
      	          else if (cols[indx] == row)
      	          {
      	             if (diag_j[pos_diag] != row_local)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
      	                break;
      	             }
      	             diag_data[pos_diag] = values[indx];
      	          }
                  else  /* insert into diag */
                  {
      	             for (j=pos_diag; j < len_diag; j++)
      	             {
      	                if (diag_j[j] == (cols[indx]-col_0))
      	                {
                           diag_data[j] = values[indx];
      		           not_found = 0;
      		           break;
      	                }
      	             }
      	             if (not_found)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
      	                break;
      	             }
                  }
                  indx++;
               }
            }
         /* processor does not own the row */
            else /*search for previous occurrences and cancel them */
	    {

   	       if (aux_matrix)
               {
                  col_indx = 0;
                  for (i=0; i < off_proc_i_indx; i=i+2)
                  {
	             row_len = off_proc_i[i+1];
	             if (off_proc_i[i] == row)
		     {
		        for (j=0; j < n; j++)
		        {
			   cnt1 = col_indx;
			   for (k=0; k < row_len; k++)
			   {
			      if (off_proc_j[cnt1] == cols[j])
			      {
                                 off_proc_j[cnt1++] = -1;
	                         offproc_cnt[my_thread_num]++;
               		         /*cancel_indx++;*/
			         /* if no repetition allowed */
                                 /* off_proc_j[col_indx] = -1;
                                    col_indx -= k;
                                 break; */
			      }
			      else
			      {
			         cnt1++;
			      }
			   }
		        }
		        col_indx += row_len;
                     }
                     else
                     {
		        col_indx += row_len;
                     }
                  }
	       }
	    } 
         } 
      } /*end parallel region */
   }
   else  /* matrix not assembled */
   {
      aux_matrix = jx_IJMatrixTranslator(matrix);
      if (aux_matrix)
      {
//         current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
         off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
         cancel_indx = jx_AuxParCSRMatrixCancelIndx(aux_matrix);
      }
      row_space = jx_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = jx_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = jx_AuxParCSRMatrixNeedAux(aux_matrix);
      if (need_aux)
      {
         aux_j = jx_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = jx_AuxParCSRMatrixAuxData(aux_matrix);
      }
      else
      {
         diag = jx_ParCSRMatrixDiag(par_matrix);
         diag_i = jx_CSRMatrixI(diag);
         diag_j = jx_CSRMatrixJ(diag);
         diag_data = jx_CSRMatrixData(diag);
         offd = jx_ParCSRMatrixOffd(par_matrix);
         offd_i = jx_CSRMatrixI(offd);
         if (num_procs > 1)
	 {
	    offd_j = jx_CSRMatrixJ(offd);
            offd_data = jx_CSRMatrixData(offd);
         }
      }
#if JX_USING_OPENMP
#pragma omp parallel 
#endif
      {
         JX_Int num_threads, my_thread_num;
         JX_Int len, rest, ns, ne;
         JX_Int *tmp_j = NULL;
         JX_Int *local_j = NULL;
         JX_Real *tmp_data = NULL;
         JX_Real *local_data = NULL;
         JX_Int tmp_indx;
         JX_Int row_len;
         JX_Int row_local;
         JX_Int i, j, k, ii, n, row;
         JX_Int not_found, size, indx, cnt1, col_indx;
         JX_Int old_size, space, cnt;

         num_threads = jx_NumActiveThreads();
         my_thread_num = jx_GetThreadNum();

         len = nrows / num_threads; 
         rest = nrows - len*num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num*(len+1);
            ne = (my_thread_num+1)*(len+1);
         }
         else
         {
            ns = my_thread_num*len+rest;
            ne = (my_thread_num+1)*len+rest;
         }

         value_start[my_thread_num] = 0;
         for (ii=ns; ii < ne; ii++)
            value_start[my_thread_num] += ncols[ii];

#if JX_USING_OPENMP
#pragma omp barrier
#endif
         if (my_thread_num == 0)
         {
            for (i=0; i < max_num_threads; i++)
               value_start[i+1] += value_start[i];
         }
#if JX_USING_OPENMP
#pragma omp barrier
#endif
         indx = 0;
         if (my_thread_num)
            indx = value_start[my_thread_num-1];
         for (ii=ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols[ii];
            /* processor owns the row */ 
            if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
            {
               row_local = row - row_partitioning[pstart]; 
               /* compute local row number */
               if (need_aux)
               {
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
   	          space = row_space[row_local]; 
   	          old_size = row_length[row_local]; 
   	          size = space - old_size;
   	          if (size < n)
      	          {
      	             size = n - size;
      	             tmp_j = jx_CTAlloc(JX_Int, size);
      	             tmp_data = jx_CTAlloc(JX_Real, size);
      	          }
      	          tmp_indx = 0;
      	          not_found = 1;
      	          size = old_size;
                  for (i=0; i < n; i++)
      	          {
      	             for (j=0; j < old_size; j++)
      	             {
      	                if (local_j[j] == cols[indx])
      	                {
                           local_data[j] = values[indx];
      		           not_found = 0;
      		           break;
      	                }
      	             }
      	             if (not_found)
      	             {
      	                if (size < space)
      	                {
      	                   local_j[size] = cols[indx];
      	                   local_data[size++] = values[indx];
      	                }
      	                else
      	                {
      	                   tmp_j[tmp_indx] = cols[indx];
      	                   tmp_data[tmp_indx++] = values[indx];
      	                }
      	             }
      	             not_found = 1;
        	     indx++;
      	          }
      	    
                  row_length[row_local] = size+tmp_indx;
                  
                  if (tmp_indx)
                  {
   	             aux_j[row_local] = jx_TReAlloc(aux_j[row_local], JX_Int, size+tmp_indx);
   	             aux_data[row_local] = jx_TReAlloc(aux_data[row_local], JX_Real, size+tmp_indx);
                     row_space[row_local] = size+tmp_indx;
                     local_j = aux_j[row_local];
                     local_data = aux_data[row_local];
                  }
   
   	          cnt = size; 
   
   	          for (i=0; i < tmp_indx; i++)
   	          {
   	             local_j[cnt] = tmp_j[i];
   	             local_data[cnt++] = tmp_data[i];
	          }
  
	          if (tmp_j)
	          { 
	             jx_TFree(tmp_j); 
	             jx_TFree(tmp_data); 
	          } 
               }
               else /* insert immediately into data in ParCSRMatrix structure */
               {
                  JX_Int offd_indx, diag_indx;
                  JX_Int offd_space, diag_space;
                  JX_Int cnt_diag, cnt_offd;
	          offd_indx = jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	          diag_indx = jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
	          cnt_diag = diag_indx;
	          cnt_offd = offd_indx;
	          diag_space = diag_i[row_local+1];
	          offd_space = offd_i[row_local+1];
	          not_found = 1;
  	          for (i=0; i < n; i++)
	          {
	             if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */	
	             {
	                for (j=offd_i[row_local]; j < offd_indx; j++)
	                {
		           if (offd_j[j] == cols[indx])
		           {
                              offd_data[j] = values[indx];
		              not_found = 0;
		              break;
		           }
	                }
	                if (not_found)
	                { 
	                   if (cnt_offd < offd_space) 
	                   { 
	                      offd_j[cnt_offd] = cols[indx];
	                      offd_data[cnt_offd++] = values[indx];
	                   } 
	                   else 
	 	           {
                              jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                              error_flag++;
	    	              if (print_level)
                                 jx_printf("Error in row %d ! Too many elements!\n", row);
                              break;
	 	           }
	                }  
	                not_found = 1;
	             }
	             else  /* insert into diag */
	             {
	                for (j=diag_i[row_local]; j < diag_indx; j++)
	                {
		           if (diag_j[j] == cols[indx])
		           {
                              diag_data[j] = values[indx];
		              not_found = 0;
		              break;
		           }
	                } 
	                if (not_found)
	                { 
	                   if (cnt_diag < diag_space) 
	                   { 
	                      diag_j[cnt_diag] = cols[indx];
	                      diag_data[cnt_diag++] = values[indx];
	                   } 
	                   else 
	 	           {
                              jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                              error_flag++;
	    	              if (print_level)
                                 jx_printf("Error in row %d ! Too many elements !\n", row);
                              break;
	 	           }
	                } 
	                not_found = 1;
	             }
	             indx++;
	          }

                  jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
                  jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

               }
            }

            /* processor does not own the row */
            else
	    {
               indx +=  n;
   	       if (aux_matrix)
               {
                  col_indx = 0;
                  for (i=0; i < off_proc_i_indx; i=i+2)
                  {
	             row_len = off_proc_i[i+1];
	             if (off_proc_i[i] == row)
		     {
		        for (j=0; j < n; j++)
		        {
			   cnt1 = col_indx;
			   for (k=0; k < row_len; k++)
			   {
			      if (off_proc_j[cnt1] == cols[j])
			      {
                                 off_proc_j[cnt1++] = -1;
               		         /*cancel_indx++;*/
	                         offproc_cnt[my_thread_num]++;
			         /* if no repetition allowed */
                                 /* off_proc_j[col_indx] = -1;
                                 col_indx -= k;
                                 break; */
			      }
			      else
			      {
			         cnt1++;
			      }
			   }
		        }
		        col_indx += row_len;
                     }
                     else
                     {
		        col_indx += row_len;
                     }
                  }
               }
	    }
         }
      } /* end parallel region */
   }
   if (error_flag) return jx_error_flag;
   if (aux_matrix)
   {
      for (i1=0; i1 < max_num_threads; i1++)
         cancel_indx += offproc_cnt[i1];
      jx_AuxParCSRMatrixCancelIndx(aux_matrix) = cancel_indx;
   }
   jx_TFree(value_start);
   jx_TFree(offproc_cnt);
   return jx_error_flag;
}

JX_Int
jx_IJMatrixSetValuesParCSR( jx_IJMatrix  *matrix,
                            JX_Int           nrows,
                            JX_Int          *ncols,
                            const JX_Int    *rows,
                            const JX_Int    *cols,
                            const JX_Real *values )
{
   jx_ParCSRMatrix *par_matrix;
   jx_CSRMatrix *diag, *offd;
   jx_AuxParCSRMatrix *aux_matrix;
   JX_Int *row_partitioning;
   JX_Int *col_partitioning;
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   JX_Int num_procs, my_id;
   JX_Int row_local, row;
   JX_Int row_len;
   JX_Int col_0, col_n;
   JX_Int i, ii, j, k, n, not_found;
   JX_Int col_indx, cancel_indx, cnt1;
   JX_Int **aux_j;
   JX_Int *local_j;
   JX_Int *tmp_j;
   JX_Real **aux_data;
   JX_Real  *local_data;
   JX_Real  *tmp_data = NULL;
   JX_Int diag_space, offd_space;
   JX_Int *row_length, *row_space;
   JX_Int need_aux;
   JX_Int tmp_indx, indx;
   JX_Int space, size, old_size;
   JX_Int cnt, cnt_diag, cnt_offd;
   JX_Int pos_diag, pos_offd;
   JX_Int len_diag, len_offd;
   JX_Int offd_indx, diag_indx;
   JX_Int *diag_i;
   JX_Int *diag_j;
   JX_Real *diag_data;
   JX_Int *offd_i;
   JX_Int *offd_j = NULL;
   JX_Real *offd_data = NULL;
   JX_Int first, pstart;
   //JX_Int current_num_elmts;
   /*JX_Int max_off_proc_elmts;*/
   JX_Int off_proc_i_indx;
   JX_Int *off_proc_i;
   JX_Int *off_proc_j;
   JX_Int print_level = jx_IJMatrixPrintLevel(matrix);
   /*JX_Real *off_proc_data;*/
   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   par_matrix = jx_IJMatrixObject( matrix );
   row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   col_partitioning = jx_IJMatrixColPartitioning(matrix);

#ifdef JX_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first =  jx_IJMatrixGlobalFirstCol(matrix);
   pstart = 0;
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
   pstart = my_id;
#endif
   if (nrows < 0)
   {
      jx_error_in_arg(2);
      if (print_level)
         jx_printf("Error! nrows negative! JX_IJMatrixSetValues\n");
   }

   if (jx_IJMatrixAssembleFlag(matrix))  /* matrix already assembled*/
   {
      JX_Int *col_map_offd = NULL;
      JX_Int num_cols_offd;
      JX_Int j_offd;
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];

         /* processor owns the row */ 
         if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
         {
            row_local = row - row_partitioning[pstart];

            /* compute local row number */
            diag = jx_ParCSRMatrixDiag(par_matrix);
            diag_i = jx_CSRMatrixI(diag);
            diag_j = jx_CSRMatrixJ(diag);
            diag_data = jx_CSRMatrixData(diag);
            offd = jx_ParCSRMatrixOffd(par_matrix);
            offd_i = jx_CSRMatrixI(offd);
            num_cols_offd = jx_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = jx_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = jx_CSRMatrixJ(offd);
               offd_data = jx_CSRMatrixData(offd);
            }
            size = diag_i[row_local+1] - diag_i[row_local] + offd_i[row_local+1] - offd_i[row_local];
      
            if (n > size)
            {
               jx_error(JX_ERROR_GENERIC);
      	       if (print_level) jx_printf (" row %d too long! \n", row);
               return jx_error_flag;
            }

            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local+1];
            len_offd = offd_i[row_local+1];
            not_found = 1;
      	
            for (i=0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
                  /* insert into offd */	
               {
      	          j_offd = jx_BinarySearch(col_map_offd,cols[indx]-first,
                                              num_cols_offd);
      	          if (j_offd == -1)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
                     return jx_error_flag;
      	          }
      	          for (j=pos_offd; j < len_offd; j++)
      	          {
      	             if (offd_j[j] == j_offd)
      	             {
                        offd_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
                     return jx_error_flag;
      	          }
      	          not_found = 1;
               }
               /* diagonal element */
      	       else if (cols[indx] == row)
      	       {
      	          if (diag_j[pos_diag] != row_local)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
      	             /* return -1;*/
                     return jx_error_flag;
      	          }
      	          diag_data[pos_diag] = values[indx];
      	       }
               else  /* insert into diag */
               {
      	          for (j=pos_diag; j < len_diag; j++)
      	          {
      	             if (diag_j[j] == (cols[indx]-col_0))
      	             {
                        diag_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf (" Error, element %d %d does not exist\n", row, cols[indx]);
      	             /* return -1; */
                     return jx_error_flag;
      	          }
               }
               indx++;
            }
         }
         
         /* processor does not own the row */  
        
         else /*search for previous occurrences and cancel them */
	 {
            aux_matrix = jx_IJMatrixTranslator(matrix);
   	    if (aux_matrix)
            {
   	       //current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	       off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	       off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
               col_indx = 0;
               cancel_indx = jx_AuxParCSRMatrixCancelIndx(aux_matrix);
               for (i=0; i < off_proc_i_indx; i=i+2)
               {
	          row_len = off_proc_i[i+1];
	          if (off_proc_i[i] == row)
		  {
		     for (j=0; j < n; j++)
		     {
			cnt1 = col_indx;
			for (k=0; k < row_len; k++)
			{
			   if (off_proc_j[cnt1] == cols[j])
			   {
                              off_proc_j[cnt1++] = -1;
               		      cancel_indx++;
			      /* if no repetition allowed */
                              /* off_proc_j[col_indx] = -1;
                                 col_indx -= k;
                                 break; */
			   }
			   else
			   {
			      cnt1++;
			   }
			}
		     }
		     col_indx += row_len;
                  }
                  else
                  {
		     col_indx += row_len;
                  }
               }
               jx_AuxParCSRMatrixCancelIndx(aux_matrix) = cancel_indx;
	    }
	 } 
      } 
   }
   else
   {
      aux_matrix = jx_IJMatrixTranslator(matrix);
      row_space = jx_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = jx_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = jx_AuxParCSRMatrixNeedAux(aux_matrix);
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
         /* processor owns the row */ 
         if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
         {
            row_local = row - row_partitioning[pstart]; 
            /* compute local row number */
            if (need_aux)
            {
               aux_j = jx_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = jx_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
   	       space = row_space[row_local]; 
   	       old_size = row_length[row_local]; 
   	       size = space - old_size;
   	       if (size < n)
      	       {
      	          size = n - size;
      	          tmp_j = jx_CTAlloc(JX_Int, size);
      	          tmp_data = jx_CTAlloc(JX_Real, size);
      	       }
      	       else
      	       {
      	          tmp_j = NULL;
      	       }
      	       tmp_indx = 0;
      	       not_found = 1;
      	       size = old_size;
               for (i=0; i < n; i++)
      	       {
      	          for (j=0; j < old_size; j++)
      	          {
      	             if (local_j[j] == cols[indx])
      	             {
                        local_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             if (size < space)
      	             {
      	                local_j[size] = cols[indx];
      	                local_data[size++] = values[indx];
      	             }
      	             else
      	             {
      	                tmp_j[tmp_indx] = cols[indx];
      	                tmp_data[tmp_indx++] = values[indx];
      	             }
      	          }
      	          not_found = 1;
        	  indx++;
      	       }
      	    
               row_length[row_local] = size+tmp_indx;
               
               if (tmp_indx)
               {
   	          aux_j[row_local] = jx_TReAlloc(aux_j[row_local], JX_Int, size+tmp_indx);
   	          aux_data[row_local] = jx_TReAlloc(aux_data[row_local], JX_Real, size+tmp_indx);
                  row_space[row_local] = size+tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }
   
   	       cnt = size; 
   
   	       for (i=0; i < tmp_indx; i++)
   	       {
   	          local_j[cnt] = tmp_j[i];
   	          local_data[cnt++] = tmp_data[i];
	       }
  
	       if (tmp_j)
	       { 
	          jx_TFree(tmp_j); 
	          jx_TFree(tmp_data); 
	       } 
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
	       offd_indx = jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	       diag_indx = jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = jx_ParCSRMatrixDiag(par_matrix);
               diag_i = jx_CSRMatrixI(diag);
               diag_j = jx_CSRMatrixJ(diag);
               diag_data = jx_CSRMatrixData(diag);
               offd = jx_ParCSRMatrixOffd(par_matrix);
               offd_i = jx_CSRMatrixI(offd);
               if (num_procs > 1)
	       {
	          offd_j = jx_CSRMatrixJ(offd);
                  offd_data = jx_CSRMatrixData(offd);
               }
	       cnt_diag = diag_indx;
	       cnt_offd = offd_indx;
	       diag_space = diag_i[row_local+1];
	       offd_space = offd_i[row_local+1];
	       not_found = 1;
  	       for (i=0; i < n; i++)
	       {
	          if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */	
	          {
	             for (j=offd_i[row_local]; j < offd_indx; j++)
	             {
		        if (offd_j[j] == cols[indx])
		        {
                           offd_data[j] = values[indx];
		           not_found = 0;
		           break;
		        }
	             }
	             if (not_found)
	             { 
	                if (cnt_offd < offd_space) 
	                { 
	                   offd_j[cnt_offd] = cols[indx];
	                   offd_data[cnt_offd++] = values[indx];
	                } 
	                else 
	 	        {
                           jx_error(JX_ERROR_GENERIC);
	    	           if (print_level)
                              jx_printf("Error in row %d ! Too many elements!\n", row);
	    	           /* return 1; */
                           return jx_error_flag;
	 	        }
	             }  
	             not_found = 1;
	          }
	          else  /* insert into diag */
	          {
	             for (j=diag_i[row_local]; j < diag_indx; j++)
	             {
		        if (diag_j[j] == cols[indx])
		        {
                           diag_data[j] = values[indx];
		           not_found = 0;
		           break;
		        }
	             } 
	             if (not_found)
	             { 
	                if (cnt_diag < diag_space) 
	                { 
	                   diag_j[cnt_diag] = cols[indx];
	                   diag_data[cnt_diag++] = values[indx];
	                } 
	                else 
	 	        {
                           jx_error(JX_ERROR_GENERIC);
	    	           if (print_level)
                              jx_printf("Error in row %d ! Too many elements !\n", row);
	    	           /* return 1; */
                           return jx_error_flag;
	 	        }
	             } 
	             not_found = 1;
	          }
	          indx++;
	       }

               jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

            }
         }

         /* processor does not own the row */
         else
	 {
            indx +=  n;
	    aux_matrix = jx_IJMatrixTranslator(matrix);
   	    if (aux_matrix)
            {
   	       //current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	       off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	       off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
               col_indx = 0;
               cancel_indx = jx_AuxParCSRMatrixCancelIndx(aux_matrix);
               for (i=0; i < off_proc_i_indx; i=i+2)
               {
	          row_len = off_proc_i[i+1];
	          if (off_proc_i[i] == row)
		  {
		     for (j=0; j < n; j++)
		     {
			cnt1 = col_indx;
			for (k=0; k < row_len; k++)
			{
			   if (off_proc_j[cnt1] == cols[j])
			   {
                              off_proc_j[cnt1++] = -1;
               		      cancel_indx++;
			      /* if no repetition allowed */
                              /* off_proc_j[col_indx] = -1;
                                 col_indx -= k;
                                 break; */
			   }
			   else
			   {
			      cnt1++;
			   }
			}
		     }
		     col_indx += row_len;
                  }
                  else
                  {
		     col_indx += row_len;
                  }
               }
               jx_AuxParCSRMatrixCancelIndx(aux_matrix) = cancel_indx;
            }
	 }
      }
   }

   return jx_error_flag;
}

JX_Int
JX_IJMatrixAssemble( JX_IJMatrix matrix )
{
   jx_IJMatrix *ijmatrix = (jx_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   if ( jx_IJMatrixObjectType(ijmatrix) == JX_PARCSR )
   {
      return(jx_IJMatrixAssembleParCSR(ijmatrix));
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
jx_IJMatrixAssembleParCSR( jx_IJMatrix *matrix )
{
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   jx_ParCSRMatrix *par_matrix = jx_IJMatrixObject(matrix);
   jx_AuxParCSRMatrix *aux_matrix = jx_IJMatrixTranslator(matrix);
   JX_Int *row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   JX_Int *col_partitioning = jx_IJMatrixColPartitioning(matrix);

   jx_CSRMatrix *diag = jx_ParCSRMatrixDiag(par_matrix);
   jx_CSRMatrix *offd = jx_ParCSRMatrixOffd(par_matrix);
   JX_Int *diag_i = jx_CSRMatrixI(diag);
   JX_Int *offd_i = jx_CSRMatrixI(offd);
   JX_Int *diag_j;
   JX_Int *offd_j = NULL;
   JX_Real *diag_data;
   JX_Real *offd_data = NULL;
   JX_Int i, j, j0;
   JX_Int num_cols_offd;
   JX_Int *diag_pos;
   JX_Int *col_map_offd;
   JX_Int *row_length;
   JX_Int **aux_j;
   JX_Real **aux_data;
   JX_Int my_id, num_procs;
   JX_Int num_rows;
   JX_Int i_diag, i_offd;
   JX_Int col_0, col_n;
   JX_Int nnz_offd;
   JX_Int *aux_offd_j;
   JX_Real temp; 
#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int base = jx_IJMatrixGlobalFirstCol(matrix);
#else
   JX_Int base = col_partitioning[0];
#endif
   JX_Int off_proc_i_indx;
   JX_Int max_off_proc_elmts;
   JX_Int current_num_elmts;
   JX_Int *off_proc_i;
   JX_Int *off_proc_j;
   JX_Real *off_proc_data;
   JX_Int offd_proc_elmts;
   JX_Int new_off_proc_i_indx;
   JX_Int cancel_indx;
   JX_Int col_indx;
   JX_Int current_indx;
   JX_Int current_i;
   JX_Int row_len;
   JX_Int max_num_threads;
   JX_Int aux_flag, aux_flag_global;

   max_num_threads = jx_NumThreads();

   /* first find out if anyone has an aux_matrix, and create one if you don't
    * have one, but other procs do */
   aux_flag = 0;
   aux_flag_global = 0;
   if(aux_matrix)
   {
      aux_flag = 1;
   }
   jx_MPI_Allreduce(&aux_flag, &aux_flag_global, 1, JX_MPI_INT, MPI_SUM, comm);
   if(aux_flag_global && (!aux_flag))
   {
      jx_MPI_Comm_rank(comm, &my_id);
      num_rows = row_partitioning[my_id+1] - row_partitioning[my_id];
      jx_AuxParCSRMatrixCreate(&aux_matrix, num_rows, num_rows, NULL);
      jx_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      jx_IJMatrixTranslator(matrix) = aux_matrix;
   }

   if (aux_matrix)
   {
      /* first delete all cancelled elements */
      cancel_indx = jx_AuxParCSRMatrixCancelIndx(aux_matrix);
      if (cancel_indx)
      {
         current_num_elmts=jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
         off_proc_i=jx_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j=jx_AuxParCSRMatrixOffProcJ(aux_matrix);
         off_proc_data=jx_AuxParCSRMatrixOffProcData(aux_matrix);
         off_proc_i_indx=jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         col_indx = 0;
         current_i = 0;
         current_indx = 0;
	 new_off_proc_i_indx = off_proc_i_indx;
         for (i=0; i < off_proc_i_indx; i= i+2)
         {
            row_len = off_proc_i[i+1];
            for (j=0; j < off_proc_i[i+1]; j++)
            {
 	       if (off_proc_j[col_indx] == -1)
	       {
		  col_indx++;
		  row_len--;
		  current_num_elmts--;
               }
               else
	       {
		  off_proc_j[current_indx] = off_proc_j[col_indx];
		  off_proc_data[current_indx++] = off_proc_data[col_indx++];
	       }
            }
            if (row_len)
	    {
	       off_proc_i[current_i] = off_proc_i[i];
	       off_proc_i[current_i+1] = row_len;
	       current_i += 2;
            }
	    else
            {
	       new_off_proc_i_indx -= 2;
            }
         }
         jx_AuxParCSRMatrixOffProcIIndx(aux_matrix) = new_off_proc_i_indx;
         jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix) = current_num_elmts;
      }
      off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
      jx_MPI_Allreduce(&off_proc_i_indx, &offd_proc_elmts, 1, JX_MPI_INT, MPI_SUM, comm);
      if (offd_proc_elmts)
      {
         max_off_proc_elmts=jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
         current_num_elmts=jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
         off_proc_i=jx_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j=jx_AuxParCSRMatrixOffProcJ(aux_matrix);
         off_proc_data=jx_AuxParCSRMatrixOffProcData(aux_matrix);
         jx_IJMatrixAssembleOffProcValsParCSR(
            matrix,off_proc_i_indx, max_off_proc_elmts, current_num_elmts,
            off_proc_i, off_proc_j, off_proc_data);
      }
   }

   if (jx_IJMatrixAssembleFlag(matrix) == 0)
   {
      jx_MPI_Comm_size(comm, &num_procs); 
      jx_MPI_Comm_rank(comm, &my_id);
#ifdef JX_NO_GLOBAL_PARTITION
      num_rows = row_partitioning[1] - row_partitioning[0]; 
      col_0 = col_partitioning[0];
      col_n = col_partitioning[1]-1;
#else
      num_rows = row_partitioning[my_id+1] - row_partitioning[my_id]; 
      col_0 = col_partitioning[my_id];
      col_n = col_partitioning[my_id+1]-1;
#endif
      /* move data into ParCSRMatrix if not there already */ 
      if (jx_AuxParCSRMatrixNeedAux(aux_matrix))
      {
         JX_Int *diag_array, *offd_array;
         diag_array = jx_CTAlloc(JX_Int, max_num_threads);
         offd_array = jx_CTAlloc(JX_Int, max_num_threads);
         aux_j = jx_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = jx_AuxParCSRMatrixAuxData(aux_matrix);
         row_length = jx_AuxParCSRMatrixRowLength(aux_matrix);
         diag_pos = jx_CTAlloc(JX_Int, num_rows);
         i_diag = 0;
         i_offd = 0;
#if JX_USING_OPENMP
#pragma omp parallel private(i,j,i_diag,i_offd)
#endif
         {
          JX_Int *local_j;
          JX_Real *local_data;
          JX_Int rest, size, ns, ne;
          JX_Int num_threads, my_thread_num;
          num_threads = jx_NumActiveThreads();
          my_thread_num = jx_GetThreadNum();

          size = num_rows/num_threads;
          rest = num_rows - size*num_threads;
        
          if (my_thread_num < rest)
          {
             ns = my_thread_num*(size + 1); 
             ne = (my_thread_num+1)*(size + 1); 
          } 
          else
          {
             ns = my_thread_num*size + rest; 
             ne = (my_thread_num+1)*size + rest; 
          } 

          i_diag = 0;
          i_offd = 0;
          for (i=ns; i < ne; i++)
          {
   	    local_j = aux_j[i];
   	    local_data = aux_data[i];
   	    diag_pos[i] = -1;
   	    for (j=0; j < row_length[i]; j++)
   	    {
   	       if (local_j[j] < col_0 || local_j[j] > col_n)
   	          i_offd++;
   	       else
   	       {
   	          i_diag++;
   	          if (local_j[j]-col_0 == i) diag_pos[i] = j;
   	       }
   	    }
          }
   	  diag_array[my_thread_num] = i_diag;
   	  offd_array[my_thread_num] = i_offd;
#if JX_USING_OPENMP
#pragma omp barrier
#endif
          if (my_thread_num == 0)
          {
            i_diag = 0;
            i_offd = 0;
            for (i = 0; i < num_threads; i++)
	    {
	       i_diag += diag_array[i];
	       i_offd += offd_array[i];
	       diag_array[i] = i_diag;
	       offd_array[i] = i_offd;
	    }
            diag_i[num_rows] = i_diag;         
            offd_i[num_rows] = i_offd;         
            if (jx_CSRMatrixJ(diag))
               jx_TFree(jx_CSRMatrixJ(diag));
            if (jx_CSRMatrixData(diag))
               jx_TFree(jx_CSRMatrixData(diag));
            if (jx_CSRMatrixJ(offd))
               jx_TFree(jx_CSRMatrixJ(offd));
            if (jx_CSRMatrixData(offd))
               jx_TFree(jx_CSRMatrixData(offd));
            diag_j = jx_CTAlloc(JX_Int, i_diag);
            diag_data = jx_CTAlloc(JX_Real, i_diag);
            if (i_offd > 0)
            {
    	       offd_j = jx_CTAlloc(JX_Int,i_offd);
               offd_data = jx_CTAlloc(JX_Real,i_offd);
            }
          }
#if JX_USING_OPENMP
#pragma omp barrier
#endif
          if (my_thread_num)
          {
             i_diag = diag_array[my_thread_num-1];
             i_offd = offd_array[my_thread_num-1];
          }
          else
          {
             i_diag = 0;
             i_offd = 0;
          }
          for (i=ns; i < ne; i++)
          {
   	    diag_i[i] = i_diag;
   	    offd_i[i] = i_offd;
   	    local_j = aux_j[i];
   	    local_data = aux_data[i];
            if (diag_pos[i] > -1)
            {
   	       diag_j[i_diag] = local_j[diag_pos[i]] - col_0;
               diag_data[i_diag++] = local_data[diag_pos[i]];
            }
   	    for (j=0; j < row_length[i]; j++)
   	    {
   	       if (local_j[j] < col_0 || local_j[j] > col_n)
   	       {
   	          offd_j[i_offd] = local_j[j];
   	          offd_data[i_offd++] = local_data[j];
   	       }
   	       else if (j != diag_pos[i])
   	       {
   	          diag_j[i_diag] = local_j[j] - col_0;
   	          diag_data[i_diag++] = local_data[j];
   	       }
   	    }
          }
         } /* end parallel region */

         jx_TFree(diag_array);
         jx_TFree(offd_array);

         jx_CSRMatrixJ(diag) = diag_j;      
         jx_CSRMatrixData(diag) = diag_data;      
         jx_CSRMatrixNumNonzeros(diag) = diag_i[num_rows];      
         if (offd_i[num_rows] > 0)
         {
            jx_CSRMatrixJ(offd) = offd_j;      
            jx_CSRMatrixData(offd) = offd_data;      
         }
         jx_CSRMatrixNumNonzeros(offd) = offd_i[num_rows];      
         jx_TFree(diag_pos);
      }
      else
      {
         /* move diagonal element into first space */
         diag_j = jx_CSRMatrixJ(diag);
         diag_data = jx_CSRMatrixData(diag);
#if JX_USING_OPENMP
#pragma omp parallel for private (i,j,j0,temp)
#endif
         for (i=0; i < num_rows; i++)
         {
   	    j0 = diag_i[i];
   	    for (j=j0; j < diag_i[i+1]; j++)
   	    {
   	       diag_j[j] -= col_0;
   	       if (diag_j[j] == i)
   	       {
   	          temp = diag_data[j0];
   	          diag_data[j0] = diag_data[j];
   	          diag_data[j] = temp;
   	          diag_j[j] = diag_j[j0];
   	          diag_j[j0] = i;
   	       }
   	    }
         }
         offd_j = jx_CSRMatrixJ(offd);
      }

      /*  generate the nonzero rows inside offd and diag by calling */

      jx_CSRMatrixSetRownnz(diag);
      jx_CSRMatrixSetRownnz(offd);

      /*  generate col_map_offd */
      nnz_offd = offd_i[num_rows];
      if (nnz_offd)
      {
         aux_offd_j = jx_CTAlloc(JX_Int, nnz_offd);
         for (i=0; i < nnz_offd; i++)
            aux_offd_j[i] = offd_j[i];
         jx_qsort0(aux_offd_j,0,nnz_offd-1);
         num_cols_offd = 1;
         for (i=0; i < nnz_offd-1; i++)
         {
            if (aux_offd_j[i+1] > aux_offd_j[i])
               aux_offd_j[num_cols_offd++] = aux_offd_j[i+1];
         }
         col_map_offd = jx_CTAlloc(JX_Int,num_cols_offd);
         for (i=0; i < num_cols_offd; i++)
   	    col_map_offd[i] = aux_offd_j[i];
#if JX_USING_OPENMP
#pragma omp parallel for private(i)
#endif
         for (i=0; i < nnz_offd; i++)
            offd_j[i] = jx_BinarySearch(col_map_offd,offd_j[i],num_cols_offd);

 	 if (base)
 	 {
	    for (i=0; i < num_cols_offd; i++)
	       col_map_offd[i] -= base;
	 } 
         jx_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;    
         jx_CSRMatrixNumCols(offd) = num_cols_offd;    
         jx_TFree(aux_offd_j);
      }
      jx_AuxParCSRMatrixDestroy(aux_matrix);
      jx_IJMatrixTranslator(matrix) = NULL;
      jx_IJMatrixAssembleFlag(matrix) = 1;
   }

   return jx_error_flag;
}

#ifndef JX_NO_GLOBAL_PARTITION

JX_Int
jx_IJMatrixAssembleOffProcValsParCSR( jx_IJMatrix *matrix,
                                      JX_Int          off_proc_i_indx,
                                      JX_Int          max_off_proc_elmts,
                                      JX_Int          current_num_elmts,
                                      JX_Int         *off_proc_i,
                                      JX_Int         *off_proc_j,
                                      JX_Real      *off_proc_data )
{
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   MPI_Request *requests = NULL;
   MPI_Status *status = NULL;
   JX_Int i, ii, j, j2, jj, n, row;
   JX_Int iii, iid, indx, ip;
   JX_Int proc_id, num_procs, my_id;
   JX_Int num_sends, num_sends3;
   JX_Int num_recvs;
   JX_Int num_requests;
   JX_Int vec_start, vec_len;
   JX_Int *send_procs;
   JX_Int *chunks;
   JX_Int *send_i;
   JX_Int *send_map_starts;
   JX_Int *dbl_send_map_starts;
   JX_Int *recv_procs;
   JX_Int *recv_chunks;
   JX_Int *recv_i;
   JX_Int *recv_vec_starts;
   JX_Int *dbl_recv_vec_starts;
   JX_Int *info;
   JX_Int *int_buffer;
   JX_Int *proc_id_mem;
   JX_Int *partitioning;
   JX_Int *displs;
   JX_Int *recv_buf;
   JX_Real *send_data;
   JX_Real *recv_data;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   partitioning = jx_IJMatrixRowPartitioning(matrix);

   info = jx_CTAlloc(JX_Int,num_procs);
   chunks = jx_CTAlloc(JX_Int,num_procs);
   proc_id_mem = jx_CTAlloc(JX_Int, off_proc_i_indx/2);
   j=0;
   for (i=0; i < off_proc_i_indx; i++)
   {
      row = off_proc_i[i++];
      if (row < 0) row = -row-1; 
      n = off_proc_i[i];
      proc_id = jx_FindProc(partitioning,row,num_procs);
      proc_id_mem[j++] = proc_id; 
      info[proc_id] += n;
      chunks[proc_id]++;
   }

   /* determine send_procs and amount of data to be sent */
   num_sends = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         num_sends++;
      }
   }
   send_procs =  jx_CTAlloc(JX_Int,num_sends);
   send_map_starts =  jx_CTAlloc(JX_Int,num_sends+1);
   dbl_send_map_starts =  jx_CTAlloc(JX_Int,num_sends+1);
   num_sends3 = 3*num_sends;
   int_buffer =  jx_CTAlloc(JX_Int,3*num_sends);
   j = 0;
   j2 = 0;
   send_map_starts[0] = 0;
   dbl_send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         send_procs[j++] = i;
         send_map_starts[j] = send_map_starts[j-1]+2*chunks[i]+info[i];
         dbl_send_map_starts[j] = dbl_send_map_starts[j-1]+info[i];
         int_buffer[j2++] = i;
	 int_buffer[j2++] = chunks[i];
	 int_buffer[j2++] = info[i];
      }
   }

   jx_TFree(chunks);

   jx_MPI_Allgather(&num_sends3,1,JX_MPI_INT,info,1,JX_MPI_INT,comm);

   displs = jx_CTAlloc(JX_Int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
      displs[i] = displs[i-1]+info[i-1];
   recv_buf = jx_CTAlloc(JX_Int, displs[num_procs]);

   jx_MPI_Allgatherv(int_buffer,num_sends3,JX_MPI_INT,recv_buf,info,displs,JX_MPI_INT,comm);

   jx_TFree(int_buffer);
   jx_TFree(info);

   /* determine recv procs and amount of data to be received */
   num_recvs = 0;
   for (j=0; j < displs[num_procs]; j+=3)
   {
      if (recv_buf[j] == my_id)
	 num_recvs++;
   }

   recv_procs = jx_CTAlloc(JX_Int,num_recvs);
   recv_chunks = jx_CTAlloc(JX_Int,num_recvs);
   recv_vec_starts = jx_CTAlloc(JX_Int,num_recvs+1);
   dbl_recv_vec_starts = jx_CTAlloc(JX_Int,num_recvs+1);

   j2 = 0;
   recv_vec_starts[0] = 0;
   dbl_recv_vec_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      for (j=displs[i]; j < displs[i+1]; j+=3)
      {
         if (recv_buf[j] == my_id)
         {
	    recv_procs[j2] = i;
	    recv_chunks[j2++] = recv_buf[j+1];
	    recv_vec_starts[j2] = recv_vec_starts[j2-1]+2*recv_buf[j+1]+recv_buf[j+2];
	    dbl_recv_vec_starts[j2] = dbl_recv_vec_starts[j2-1]+recv_buf[j+2];
         }
         if (j2 == num_recvs) break;
      }
   }
   jx_TFree(recv_buf);
   jx_TFree(displs);

   /* set up data to be sent to send procs */
   /* send_i contains for each send proc : row no., no. of elmts and column
      indices, send_data contains corresponding values */
      
   send_i = jx_CTAlloc(JX_Int,send_map_starts[num_sends]);
   send_data = jx_CTAlloc(JX_Real,dbl_send_map_starts[num_sends]);
   recv_i = jx_CTAlloc(JX_Int,recv_vec_starts[num_recvs]);
   recv_data = jx_CTAlloc(JX_Real,dbl_recv_vec_starts[num_recvs]);
    
   j=0;
   jj=0;
   for (i=0; i < off_proc_i_indx; i++)
   {
      row = off_proc_i[i++]; 
      n = off_proc_i[i];
      proc_id = proc_id_mem[i/2];
      indx = jx_BinarySearch(send_procs,proc_id,num_sends);
      iii = send_map_starts[indx];
      iid = dbl_send_map_starts[indx];
      send_i[iii++] = row;
      send_i[iii++] = n;
      for (ii = 0; ii < n; ii++)
      {
         send_i[iii++] = off_proc_j[jj];
         send_data[iid++] = off_proc_data[jj++];
      }
      send_map_starts[indx] = iii;
      dbl_send_map_starts[indx] = iid;
   }

   jx_TFree(proc_id_mem);

   for (i=num_sends; i > 0; i--)
   {
      send_map_starts[i] = send_map_starts[i-1];
      dbl_send_map_starts[i] = dbl_send_map_starts[i-1];
   }
   send_map_starts[0] = 0;
   dbl_send_map_starts[0] = 0;

   num_requests = num_recvs+num_sends;

   requests = jx_CTAlloc(MPI_Request, num_requests);
   status = jx_CTAlloc(MPI_Status, num_requests);

   j=0; 
   for (i=0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i+1] - vec_start;
      ip = recv_procs[i];
      jx_MPI_Irecv(&recv_i[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i+1] - vec_start;
      ip = send_procs[i];
      jx_MPI_Isend(&send_i[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
   }

   if (num_requests)
   {
      jx_MPI_Waitall(num_requests, requests, status);
   }

   j=0;
   for (i=0; i < num_recvs; i++)
   {
      vec_start = dbl_recv_vec_starts[i];
      vec_len = dbl_recv_vec_starts[i+1] - vec_start;
      ip = recv_procs[i];
      jx_MPI_Irecv(&recv_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
      vec_start = dbl_send_map_starts[i];
      vec_len = dbl_send_map_starts[i+1] - vec_start;
      ip = send_procs[i];
      jx_MPI_Isend(&send_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
   }

   if (num_requests)
   {
      jx_MPI_Waitall(num_requests, requests, status);
   }

   jx_TFree(requests);
   jx_TFree(status);
   jx_TFree(send_i);
   jx_TFree(send_data);
   jx_TFree(send_procs);
   jx_TFree(send_map_starts);
   jx_TFree(dbl_send_map_starts);
   jx_TFree(recv_procs);
   jx_TFree(recv_vec_starts);
   jx_TFree(dbl_recv_vec_starts);

   j = 0;
   j2 = 0;
   for (i=0; i < num_recvs; i++)
   {
      for (ii=0; ii < recv_chunks[i]; ii++)
      {
         row = recv_i[j];
 	 jx_IJMatrixAddToValuesParCSR(matrix,1,&recv_i[j+1],&row,&recv_i[j+2],&recv_data[j2]);
	 j2 += recv_i[j+1]; 
	 j += recv_i[j+1]+2; 
      }
   }
   jx_TFree(recv_chunks);
   jx_TFree(recv_i);
   jx_TFree(recv_data);

   return jx_error_flag;
}

#else

/* assumed partition version */

JX_Int
jx_IJMatrixAssembleOffProcValsParCSR( jx_IJMatrix *matrix,
                                      JX_Int          off_proc_i_indx,
                                      JX_Int          max_off_proc_elmts,
                                      JX_Int          current_num_elmts,
                                      JX_Int         *off_proc_i,
                                      JX_Int         *off_proc_j,
                                      JX_Real      *off_proc_data )
{
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   JX_Int i, j, k, in_i;
   JX_Int myid;
   JX_Int proc_id, last_proc, prev_id, tmp_id;
   JX_Int max_response_size;
   JX_Int global_num_cols;
   JX_Int global_first_col;
   JX_Int global_first_row;
   JX_Int ex_num_contacts = 0, num_rows = 0;
   JX_Int range_start, range_end;
   JX_Int num_elements;
   JX_Int storage;
   JX_Int indx;
   JX_Int row, num_ranges;
   JX_Int num_recvs;
   JX_Int counter, upper_bound;
   JX_Int num_real_procs;
   JX_Int current_proc, original_proc_indx;
   JX_Int *row_list=NULL, *row_list_num_elements=NULL;
   JX_Int *a_proc_id=NULL, *orig_order=NULL;
   JX_Int *real_proc_id = NULL, *us_real_proc_id = NULL;
   JX_Int *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL, *ex_contact_buf = NULL;
   JX_Int *recv_starts=NULL;
   JX_Int *response_buf = NULL, *response_buf_starts=NULL;
   JX_Int *num_rows_per_proc = NULL, *num_elements_total = NULL;
   JX_Int *argsort_contact_procs = NULL;
   JX_Int  obj_size_bytes, int_size, complex_size;
   JX_Int  tmp_int;
   JX_Int *col_ptr;
   JX_Int *int_data = NULL;
   JX_Int int_data_size = 0, complex_data_size = 0;
   void *void_contact_buf = NULL;
   void *index_ptr;
   void *recv_data_ptr;
   JX_Real  tmp_complex;
   JX_Real *col_data_ptr;
   JX_Real *complex_data = NULL;
   jx_DataExchangeResponse  response_obj1, response_obj2;
   jx_ProcListElements      send_proc_obj;
   jx_IJAssumedPart   *apart;

   jx_MPI_Comm_rank(comm, &myid);
   global_num_cols = jx_IJMatrixGlobalNumCols(matrix);
   global_first_col = jx_IJMatrixGlobalFirstCol(matrix);
   global_first_row = jx_IJMatrixGlobalFirstRow(matrix);

   num_rows = off_proc_i_indx/2;
   
   /* verify that we have created the assumed partition */
   if  (jx_IJMatrixAssumedPart(matrix) == NULL)
   {
      jx_IJMatrixCreateAssumedPartition(matrix);
   }

   apart = jx_IJMatrixAssumedPart(matrix);

   /*if  (jx_ParCSRMatrixAssumedPartition(par_matrix) == NULL)
   {
      jx_ParCSRMatrixCreateAssumedPartition(par_matrix);
   }

   apart = jx_ParCSRMatrixAssumedPartition(par_matrix);*/

   row_list = jx_CTAlloc(JX_Int, num_rows);
   row_list_num_elements = jx_CTAlloc(JX_Int, num_rows);
   a_proc_id = jx_CTAlloc(JX_Int, num_rows);
   orig_order =  jx_CTAlloc(JX_Int, num_rows);
   real_proc_id = jx_CTAlloc(JX_Int, num_rows);

   /* get the assumed processor id for each row */
   if (num_rows > 0 )
   {
      for (i=0; i < num_rows; i++)
      {
         row = off_proc_i[i*2];
         if (row < 0) row = -row-1; 
         row_list[i] = row;
         row_list_num_elements[i] = off_proc_i[i*2+1];
         jx_GetAssumedPartitionProcFromRow(comm, row, global_first_row, global_num_cols, &proc_id);
         a_proc_id[i] = proc_id;
         orig_order[i] = i;
      }

      /* now we need to find the actual order of each row  - sort on row -
         this will result in proc ids sorted also...*/

      jx_qsort3i(row_list, a_proc_id, orig_order, 0, num_rows -1);

      /* calculate the number of contacts */
      ex_num_contacts = 1;
      last_proc = a_proc_id[0];
      for (i=1; i < num_rows; i++)
      {
         if (a_proc_id[i] > last_proc)      
         {
            ex_num_contacts++;
            last_proc = a_proc_id[i];
         }
      }
      
   } 
   
   /* now we will go through a create a contact list - need to contact assumed
      processors and find out who the actual row owner is - we will contact with
      a range (2 numbers) */

   ex_contact_procs = jx_CTAlloc(JX_Int, ex_num_contacts);
   ex_contact_vec_starts =  jx_CTAlloc(JX_Int, ex_num_contacts+1);
   ex_contact_buf =  jx_CTAlloc(JX_Int, ex_num_contacts*2);

   counter = 0;
   range_end = -1;
   for (i=0; i< num_rows; i++) 
   {
      if (row_list[i] > range_end)
      {
         /* assumed proc */
         proc_id = a_proc_id[i];

         /* end of prev. range */
         if (counter > 0)  ex_contact_buf[counter*2 - 1] = row_list[i-1];
         
         /*start new range*/
    	 ex_contact_procs[counter] = proc_id;
         ex_contact_vec_starts[counter] = counter*2;
         ex_contact_buf[counter*2] =  row_list[i];
         counter++;
         
         jx_GetAssumedPartitionRowRange(comm, proc_id, global_first_col, global_num_cols, 
                                           &range_start, &range_end); 
      }
   }
   /*finish the starts*/
   ex_contact_vec_starts[counter] =  counter*2;
   /*finish the last range*/
   if (counter > 0)  
      ex_contact_buf[counter*2 - 1] = row_list[num_rows - 1];

   /*don't allocate space for responses */
    
   /* create response object - can use same fill response as used in the commpkg
      routine */
   response_obj1.fill_response = jx_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/ 
   response_obj1.data2 = NULL;
   
   max_response_size = 6;  /* 6 means we can fit 3 ranges*/
   
   jx_DataExchangeList(ex_num_contacts, ex_contact_procs, 
                          ex_contact_buf, ex_contact_vec_starts, sizeof(JX_Int), 
                          sizeof(JX_Int), &response_obj1, max_response_size, 1, 
                          comm, (void**) &response_buf, &response_buf_starts);

   /* now response_buf contains a proc_id followed by a range upper bound */

   jx_TFree(ex_contact_procs);
   jx_TFree(ex_contact_buf);
   jx_TFree(ex_contact_vec_starts);

   jx_TFree(a_proc_id);

   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];   
   num_ranges = num_ranges/2;

   prev_id = -1;
   j = 0;
   counter = 0;
   num_real_procs = 0;
  
   /* loop through ranges - create a list of actual processor ids*/
   for (i=0; i<num_ranges; i++)
   {
      upper_bound = response_buf[i*2+1];
      counter = 0;
      tmp_id = response_buf[i*2];

      /* loop through row_list entries - counting how many are in the range */
      while (j < num_rows && row_list[j] <= upper_bound)    
      {
         real_proc_id[j] = tmp_id;
         j++;
         counter++;       
      }
      if (counter > 0 && tmp_id != prev_id)        
      {
         num_real_procs++;
      }

      prev_id = tmp_id;
   }

   /* now we have the list of real procesors ids (real_proc_id) - and the number
      of distinct ones - so now we can set up data to be sent - we have
      JX_Int data and JX_Real data.  that we will need to pack
      together */
   
   /* first find out how many rows and elements we need to send per proc - so we
      can do storage */
   
   ex_contact_procs = jx_CTAlloc(JX_Int, num_real_procs);
   num_rows_per_proc = jx_CTAlloc(JX_Int, num_real_procs);
   num_elements_total  =  jx_CTAlloc(JX_Int, num_real_procs); 
   
   counter = 0;
   
   if (num_real_procs > 0 )
   {
      ex_contact_procs[0] = real_proc_id[0];
      num_rows_per_proc[0] = 1;
      num_elements_total[0] = row_list_num_elements[orig_order[0]];

      /* loop through real procs - these are sorted (row_list is sorted also)*/
      for (i=1; i < num_rows; i++)
      {
         if (real_proc_id[i] == ex_contact_procs[counter]) /* same processor */
         {
            num_rows_per_proc[counter] += 1; /*another row */
            num_elements_total[counter] += row_list_num_elements[orig_order[i]];
         }
         else /* new processor */
         {
            counter++;
            ex_contact_procs[counter] = real_proc_id[i];
            num_rows_per_proc[counter] = 1;
            num_elements_total[counter] = row_list_num_elements[orig_order[i]];
         }
      }
   }
   
   /* to pack together, we need to use the largest obj. size of
      (JX_Int) and (JX_Real) - if these are much different, then we are
      wasting some storage, but I do not think that it will be a
      large amount since this function should not be used on really
      large amounts of data anyway*/
   int_size = sizeof(JX_Int);
   complex_size = sizeof(JX_Real);
   
   obj_size_bytes = jx_max(int_size, complex_size);

   /* set up data to be sent to send procs */
   /* for each proc, ex_contact_buf contains #rows, row #,
      no. elements, col indicies, col data, row #, no. elements, col
      indicies, col data, etc. */
      
   /* first calculate total storage and make vec_starts arrays */
   storage = 0;
   ex_contact_vec_starts = jx_CTAlloc(JX_Int, num_real_procs + 1);
   ex_contact_vec_starts[0] = -1;
   
   for (i=0; i < num_real_procs; i++)
   {
      storage += 1 + 2 * num_rows_per_proc[i] + 2* num_elements_total[i];
      ex_contact_vec_starts[i+1] = -storage-1; /* need negative for next loop */
   }      

   jx_TFree(num_elements_total);

   void_contact_buf = jx_MAlloc(storage*obj_size_bytes);
   index_ptr = void_contact_buf; /* step through with this index */

   /* for each proc: #rows, row #, no. elements, 
      col indicies, col data, row #, no. elements, col indicies, col data, etc. */
      
   /* un-sort real_proc_id - we want to access data arrays in order, so 
      cheaper to do this*/
   us_real_proc_id =  jx_CTAlloc(JX_Int, num_rows);
   for (i=0; i < num_rows; i++)
   {
      us_real_proc_id[orig_order[i]] = real_proc_id[i];
   }
   jx_TFree(real_proc_id);

   counter = 0; /* index into data arrays */
   prev_id = -1;
   for (i=0; i < num_rows; i++)
   {
      proc_id = us_real_proc_id[i];
      /* can't use row list[i] - you loose the negative signs that differentiate
         add/set values */
      row = off_proc_i[i*2];
      num_elements = row_list_num_elements[i];
      /* find position of this processor */
      indx = jx_BinarySearch(ex_contact_procs, proc_id, num_real_procs);
      in_i = ex_contact_vec_starts[indx];

      index_ptr = (void *) ((char *) void_contact_buf + in_i*obj_size_bytes);

      /* first time for this processor - add the number of rows to the buffer */
      if (in_i < 0)
      {
         in_i = -in_i - 1;
         /* re-calc. index_ptr since in_i was negative */
         index_ptr = (void *) ((char *) void_contact_buf + in_i*obj_size_bytes);

         tmp_int =  num_rows_per_proc[indx];
         memcpy( index_ptr, &tmp_int, int_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);

         in_i++;
      }
      /* add row # */   
      memcpy( index_ptr, &row, int_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in_i++;
            
      /* add number of elements */   
      memcpy( index_ptr, &num_elements, int_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in_i++;

      /* now add col indices  */
      for (j=0; j< num_elements; j++)
      {
         tmp_int = off_proc_j[counter+j]; /* col number */

         memcpy( index_ptr, &tmp_int, int_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
         in_i ++;
      }

      /* now add data */
      for (j=0; j< num_elements; j++)
      {
         tmp_complex = off_proc_data[counter++]; /* value */

         memcpy(index_ptr, &tmp_complex, complex_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
         in_i++;
      }

      /* increment the indexes to keep track of where we are - we
       * adjust below to be actual starts*/
      ex_contact_vec_starts[indx] = in_i;
   }
   
   /* some clean up */
 
   jx_TFree(response_buf);
   jx_TFree(response_buf_starts);

   jx_TFree(us_real_proc_id);
   jx_TFree(orig_order);
   jx_TFree(row_list);
   jx_TFree(row_list_num_elements);
   jx_TFree(num_rows_per_proc);
   
   for (i=num_real_procs; i > 0; i--)
   {
      ex_contact_vec_starts[i] = ex_contact_vec_starts[i-1];
   }

   ex_contact_vec_starts[0] = 0;

   /* now send the data */

   /***********************************/

   /* first get the interger info in send_proc_obj */

   /* the response we expect is just a confirmation*/
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/

   /* use the send_proc_obj for the info kept from contacts */
   /*estimate inital storage allocation */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_real_procs + 5;
   send_proc_obj.id = jx_CTAlloc(JX_Int, send_proc_obj.storage_length+1);
   send_proc_obj.vec_starts = jx_CTAlloc(JX_Int, send_proc_obj.storage_length+1);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = storage + 20;
   send_proc_obj.v_elements = jx_MAlloc(obj_size_bytes*send_proc_obj.element_storage_length);

   response_obj2.fill_response = jx_FillResponseIJOffProcVals;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj;

   max_response_size = 0;

   jx_DataExchangeList(num_real_procs, ex_contact_procs, 
                          void_contact_buf, ex_contact_vec_starts, obj_size_bytes,
                          0, &response_obj2, max_response_size, 2, 
                          comm,  (void **) &response_buf, &response_buf_starts);

   jx_TFree(response_buf);
   jx_TFree(response_buf_starts);

   jx_TFree(ex_contact_procs);
   jx_TFree(void_contact_buf);
   jx_TFree(ex_contact_vec_starts);

   /* Now we can unpack the send_proc_objects and call set 
      and add to values functions.  We unpack messages in a 
      deterministic order, using processor rank */
   
   num_recvs = send_proc_obj.length; 
   argsort_contact_procs = jx_CTAlloc(JX_Int, num_recvs);
   for(i=0; i < num_recvs; i++)
   {
      argsort_contact_procs[i] = i;
   }
   /* This sort's the id array, but the original indices are stored in
    * argsort_contact_procs */
   jx_qsort2i(send_proc_obj.id, argsort_contact_procs, 0, num_recvs-1);

   /* alias */
   recv_data_ptr = send_proc_obj.v_elements;
   recv_starts = send_proc_obj.vec_starts;

   for (i=0; i < num_recvs; i++)
   {
      
      /* Find the current processor in order, and reset recv_data_ptr to that processor's message */
      original_proc_indx = argsort_contact_procs[i];
      current_proc = send_proc_obj.id[i];
      indx = recv_starts[original_proc_indx];
      recv_data_ptr = (void *) ((char *) send_proc_obj.v_elements + indx*obj_size_bytes);

      /* get the number of rows for this recv */
      memcpy( &num_rows, recv_data_ptr, int_size);
      recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
      indx++;

      for (j=0; j < num_rows; j++) /* for each row: unpack info */
      {
         /* row # */
         memcpy( &row, recv_data_ptr, int_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* num elements for this row */
         memcpy( &num_elements, recv_data_ptr, int_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* col indices */
         if (int_size == obj_size_bytes)
         {
            col_ptr = (JX_Int *) recv_data_ptr;
            recv_data_ptr =
               (void *) ((char *)recv_data_ptr + num_elements*obj_size_bytes);
         }
         else /* copy data */
         {
            if (int_data_size < num_elements)
            {
               int_data = jx_TReAlloc(int_data, JX_Int, num_elements + 10);
            }
            for (k=0; k< num_elements; k++)
            { 
               memcpy( &int_data[k], recv_data_ptr, int_size);
               recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
            }
            col_ptr = int_data;
         }
         
         /* col data */
         if (complex_size == obj_size_bytes)
         {
            col_data_ptr = (JX_Real *) recv_data_ptr;
            recv_data_ptr =
               (void *) ((char *)recv_data_ptr + num_elements*obj_size_bytes);
         }
         else /* copy data */
         {
            if (complex_data_size < num_elements)
            {
               complex_data =
                  jx_TReAlloc(complex_data, JX_Real, num_elements + 10);
            }
            for (k=0; k< num_elements; k++)
            { 
               memcpy( &complex_data[k], recv_data_ptr, complex_size);
               recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
            }
            col_data_ptr = complex_data;
         }

 	 jx_IJMatrixAddToValuesParCSR(matrix,1,&num_elements,&row,col_ptr,col_data_ptr);
         indx += (num_elements*2); 

      }
   }
   jx_TFree(send_proc_obj.v_elements);
   jx_TFree(send_proc_obj.vec_starts);
   jx_TFree(send_proc_obj.id);
   jx_TFree(argsort_contact_procs);
 
   if (int_data) jx_TFree(int_data);
   if (complex_data) jx_TFree(complex_data);

   return jx_error_flag;
}

#endif

JX_Int
jx_IJMatrixCreateAssumedPartition( jx_IJMatrix *matrix )
{
   JX_Int global_num_rows;
   JX_Int global_first_row;
   JX_Int myid;
   JX_Int  row_start = 0, row_end = 0;
   JX_Int *row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   MPI_Comm comm;
   jx_IJAssumedPart *apart;

   global_num_rows = jx_IJMatrixGlobalNumRows(matrix);
   global_first_row = jx_IJMatrixGlobalFirstRow(matrix);
   comm = jx_IJMatrixComm(matrix);

   /* find out my actual range of rows and rowumns */
   row_start = row_partitioning[0];
   row_end = row_partitioning[1]-1;
   jx_MPI_Comm_rank(comm, &myid);

   /* allocate space */
   apart = jx_CTAlloc(jx_IJAssumedPart, 1);

  /* get my assumed partitioning  - we want row partitioning of the matrix
      for off processor values - so we use the row start and end 
     Note that this is different from the assumed partitioning for the parcsr matrix
     which needs it for matvec multiplications and therefore needs to do it for
     the col partitioning */
   jx_GetAssumedPartitionRowRange( comm, myid, global_first_row, 
			global_num_rows, &(apart->row_start), &(apart->row_end));

  /*allocate some space for the partition of the assumed partition */
   apart->length = 0;

  /*room for 10 owners of the assumed partition*/ 
   apart->storage_length = 10; /*need to be >=1 */ 
   apart->proc_list = jx_TAlloc(JX_Int, apart->storage_length);
   apart->row_start_list = jx_TAlloc(JX_Int, apart->storage_length);
   apart->row_end_list = jx_TAlloc(JX_Int, apart->storage_length);

  /* now we want to reconcile our actual partition with the assumed partition */
   jx_LocateAssummedPartition(comm, row_start, row_end, global_first_row,
			global_num_rows, apart, myid);

  /* this partition will be saved in the matrix data structure until the matrix is destroyed */
   jx_IJMatrixAssumedPart(matrix) = apart;

   return jx_error_flag;
}

JX_Int
jx_FillResponseIJOffProcVals( void      *p_recv_contact_buf, 
                              JX_Int        contact_size,
                              JX_Int        contact_proc,
                              void      *ro,
                              MPI_Comm   comm,
                              void     **p_send_response_buf,
                              JX_Int       *response_message_size )
{
   JX_Int    myid;
   JX_Int    index, count, elength;
   JX_Int object_size;
   void *index_ptr;

   jx_DataExchangeResponse *response_obj = ro;

   jx_ProcListElements *send_proc_obj = response_obj->data2;

   object_size = jx_max(sizeof(JX_Int), sizeof(JX_Real));

   jx_MPI_Comm_rank(comm, &myid);

   /*check to see if we need to allocate more space in send_proc_obj for vec starts
    * and id */
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 20; /*add space for 20 more contact*/
      send_proc_obj->vec_starts = jx_TReAlloc(send_proc_obj->vec_starts,JX_Int,
                                                 send_proc_obj->storage_length + 1);
      if(send_proc_obj->id != NULL)
      {
         send_proc_obj->id = jx_TReAlloc(send_proc_obj->id, JX_Int,
                                         send_proc_obj->storage_length + 1);
      }
   }

   /*initialize*/ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /* current number of elements */
   if( send_proc_obj->id != NULL)
   {
      send_proc_obj->id[count] = contact_proc;
   }

   /*do we need more storage for the elements?*/
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = jx_max(contact_size, 100);
      elength += index;
      send_proc_obj->v_elements = jx_ReAlloc(send_proc_obj->v_elements, elength*object_size);
      send_proc_obj->element_storage_length = elength;
   }
   /*populate send_proc_obj*/
   index_ptr = (void *) ((char *) send_proc_obj->v_elements + index*object_size);

   memcpy(index_ptr, p_recv_contact_buf , object_size*contact_size);

   send_proc_obj->vec_starts[count+1] = index + contact_size;
   send_proc_obj->length++;

   /* output - no message to return (confirmation) */
  *response_message_size = 0; 

   return jx_error_flag;
}

JX_Int
jx_IJMatrixAddToValuesParCSR( jx_IJMatrix   *matrix,
                              JX_Int            nrows,
                              JX_Int           *ncols,
                              const JX_Int     *rows,
                              const JX_Int     *cols,
                              const JX_Real  *values )
{
   jx_ParCSRMatrix *par_matrix;
   jx_CSRMatrix *diag, *offd;
   jx_AuxParCSRMatrix *aux_matrix;
   JX_Int *row_partitioning;
   JX_Int *col_partitioning;
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   JX_Int num_procs, my_id;
   JX_Int row_local, row;
   JX_Int col_0, col_n;
   JX_Int i, ii, j, n, not_found;
   JX_Int **aux_j;
   JX_Int *local_j;
   JX_Int *tmp_j;
   JX_Real **aux_data;
   JX_Real  *local_data;
   JX_Real  *tmp_data = NULL;
   JX_Int diag_space, offd_space;
   JX_Int *row_length, *row_space;
   JX_Int need_aux;
   JX_Int tmp_indx, indx;
   JX_Int space, size, old_size;
   JX_Int cnt, cnt_diag, cnt_offd;
   JX_Int pos_diag, pos_offd;
   JX_Int len_diag, len_offd;
   JX_Int offd_indx, diag_indx;
   JX_Int first, pstart;
   JX_Int *diag_i;
   JX_Int *diag_j;
   JX_Real *diag_data;
   JX_Int *offd_i;
   JX_Int *offd_j = NULL;
   JX_Real *offd_data = NULL;
   JX_Int current_num_elmts;
   JX_Int max_off_proc_elmts;
   JX_Int off_proc_i_indx;
   JX_Int *off_proc_i;
   JX_Int *off_proc_j;
   JX_Real *off_proc_data;
   JX_Int print_level = jx_IJMatrixPrintLevel(matrix);

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   par_matrix = jx_IJMatrixObject( matrix );
   row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   col_partitioning = jx_IJMatrixColPartitioning(matrix);
#ifdef JX_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first = jx_IJMatrixGlobalFirstCol(matrix);
   pstart = 0;
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
   pstart = my_id;
#endif
   if (jx_IJMatrixAssembleFlag(matrix))
   {
      JX_Int num_cols_offd;
      JX_Int *col_map_offd = NULL;
      JX_Int j_offd;
      indx = 0;

      /* AB - 4/06 - need to get this object*/
      aux_matrix = jx_IJMatrixTranslator(matrix);

      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
         if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
         {
            row_local = row - row_partitioning[pstart]; 
            /* compute local row number */
            diag = jx_ParCSRMatrixDiag(par_matrix);
            diag_i = jx_CSRMatrixI(diag);
            diag_j = jx_CSRMatrixJ(diag);
            diag_data = jx_CSRMatrixData(diag);
            offd = jx_ParCSRMatrixOffd(par_matrix);
            offd_i = jx_CSRMatrixI(offd);
            num_cols_offd = jx_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = jx_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = jx_CSRMatrixJ(offd);
               offd_data = jx_CSRMatrixData(offd);
            }
            size = diag_i[row_local+1] - diag_i[row_local]
               + offd_i[row_local+1] - offd_i[row_local];
      
            if (n > size)
            {
               jx_error(JX_ERROR_GENERIC);
      	       if (print_level) jx_printf (" row %d too long! \n", row);
      	       /* return -1; */
               return jx_error_flag;
            }

            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local+1];
            len_offd = offd_i[row_local+1];
            not_found = 1;
      	
            for (i=0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
                  /* insert into offd */	
               {
      	          j_offd = jx_BinarySearch(col_map_offd,cols[indx]-first,
                                              num_cols_offd);
      	          if (j_offd == -1)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
                     return jx_error_flag;
      	             /* return -1; */
      	          }
      	          for (j=pos_offd; j < len_offd; j++)
      	          {
      	             if (offd_j[j] == j_offd)
      	             {
                        offd_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
      	             /* return -1;*/
                     return jx_error_flag;
      	          }
      	          not_found = 1;
               }
               /* diagonal element */
      	       else if (cols[indx] == row)
      	       {
      	          if (diag_j[pos_diag] != row_local)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
      	             /* return -1; */
                     return jx_error_flag;
      	          }
      	          diag_data[pos_diag] += values[indx];
      	       }
               else  /* insert into diag */
               {
      	          for (j=pos_diag; j < len_diag; j++)
      	          {
      	             if (diag_j[j] == (cols[indx]-col_0))
      	             {
                        diag_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     jx_error(JX_ERROR_GENERIC);
      	             if (print_level)
			jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
      	             /* return -1;*/
                     return jx_error_flag;
      	          }
               }
               indx++;
            }
         }
         /* not my row */
         else
	 {
   	    if (!aux_matrix)
            {
               size = row_partitioning[pstart+1]-row_partitioning[pstart];
	       jx_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
      	       jx_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      	       jx_IJMatrixTranslator(matrix) = aux_matrix;
            }
   	    current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts = jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = jx_AuxParCSRMatrixOffProcData(aux_matrix);

	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = jx_max(n,1000);
	       jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
                  max_off_proc_elmts;
   	       jx_AuxParCSRMatrixOffProcI(aux_matrix)
                  = jx_CTAlloc(JX_Int,2*max_off_proc_elmts);
   	       jx_AuxParCSRMatrixOffProcJ(aux_matrix)
                  = jx_CTAlloc(JX_Int,max_off_proc_elmts);
   	       jx_AuxParCSRMatrixOffProcData(aux_matrix)
                  = jx_CTAlloc(JX_Real,max_off_proc_elmts);
   	       off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = jx_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = jx_TReAlloc(off_proc_i,JX_Int,2*max_off_proc_elmts);
               off_proc_j = jx_TReAlloc(off_proc_j,JX_Int,max_off_proc_elmts);
               off_proc_data = jx_TReAlloc(off_proc_data,JX_Real,max_off_proc_elmts);
	       jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
	       jx_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       jx_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       jx_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }

            /* AB - 4/6 - the row should be negative to indicate an add */
            /* UMY - 12/28/09 - now positive since we eliminated the feature of
               setting on other processors */
            /* off_proc_i[off_proc_i_indx++] = row; */
            off_proc_i[off_proc_i_indx++] = row;
            
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    jx_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix) = current_num_elmts; 
	 }
      }
   }
   
   /* not assembled */
   else
   {
      aux_matrix = jx_IJMatrixTranslator(matrix);
      row_space = jx_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = jx_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = jx_AuxParCSRMatrixNeedAux(aux_matrix);
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
         if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
         {
            row_local = row - row_partitioning[pstart]; 
            /* compute local row number */
            if (need_aux)
            {
               aux_j = jx_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = jx_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
   	       space = row_space[row_local]; 
   	       old_size = row_length[row_local]; 
   	       size = space - old_size;
   	       if (size < n)
      	       {
      	          size = n - size;
      	          tmp_j = jx_CTAlloc(JX_Int,size);
      	          tmp_data = jx_CTAlloc(JX_Real,size);
      	       }
      	       else
      	       {
      	          tmp_j = NULL;
      	       }
      	       tmp_indx = 0;
      	       not_found = 1;
      	       size = old_size;
               for (i=0; i < n; i++)
      	       {
      	          for (j=0; j < old_size; j++)
      	          {
      	             if (local_j[j] == cols[indx])
      	             {
                        local_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             if (size < space)
      	             {
      	                local_j[size] = cols[indx];
      	                local_data[size++] = values[indx];
      	             }
      	             else
      	             {
      	                tmp_j[tmp_indx] = cols[indx];
      	                tmp_data[tmp_indx++] = values[indx];
      	             }
      	          }
      	          not_found = 1;
        	  indx++;
      	       }

               row_length[row_local] = size+tmp_indx;

               if (tmp_indx)
               {
   	          aux_j[row_local] = jx_TReAlloc(aux_j[row_local],JX_Int,size+tmp_indx);
   	          aux_data[row_local] = jx_TReAlloc(aux_data[row_local],JX_Real,size+tmp_indx);
                  row_space[row_local] = size+tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }

   	       cnt = size; 

   	       for (i=0; i < tmp_indx; i++)
   	       {
   	          local_j[cnt] = tmp_j[i];
   	          local_data[cnt++] = tmp_data[i];
	       }
  
	       if (tmp_j)
	       { 
	          jx_TFree(tmp_j); 
	          jx_TFree(tmp_data); 
	       } 
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
	       offd_indx = jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	       diag_indx = jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = jx_ParCSRMatrixDiag(par_matrix);
               diag_i = jx_CSRMatrixI(diag);
               diag_j = jx_CSRMatrixJ(diag);
               diag_data = jx_CSRMatrixData(diag);
               offd = jx_ParCSRMatrixOffd(par_matrix);
               offd_i = jx_CSRMatrixI(offd);
               if (num_procs > 1)
	       {
	          offd_j = jx_CSRMatrixJ(offd);
                  offd_data = jx_CSRMatrixData(offd);
               }
	       cnt_diag = diag_indx;
	       cnt_offd = offd_indx;
	       diag_space = diag_i[row_local+1];
	       offd_space = offd_i[row_local+1];
	       not_found = 1;
  	       for (i=0; i < n; i++)
	       {
	          if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */	
	          {
	             for (j=offd_i[row_local]; j < offd_indx; j++)
	             {
		        if (offd_j[j] == cols[indx])
		        {
                           offd_data[j] += values[indx];
		           not_found = 0;
		           break;
		        }
	             }
	             if (not_found)
	             { 
	                if (cnt_offd < offd_space) 
	                { 
	                   offd_j[cnt_offd] = cols[indx];
	                   offd_data[cnt_offd++] = values[indx];
	                } 
	                else 
	 	        {
                           jx_error(JX_ERROR_GENERIC);
	    	           if (print_level)
                              jx_printf("Error in row %d ! Too many elements!\n", row);
	    	           /* return 1;*/
                           return jx_error_flag;
	 	        }
	             }  
	             not_found = 1;
	          }
	          else  /* insert into diag */
	          {
	             for (j=diag_i[row_local]; j < diag_indx; j++)
	             {
		        if (diag_j[j] == cols[indx])
		        {
                           diag_data[j] += values[indx];
		           not_found = 0;
		           break;
		        }
	             } 
	             if (not_found)
	             { 
	                if (cnt_diag < diag_space) 
	                { 
	                   diag_j[cnt_diag] = cols[indx];
	                   diag_data[cnt_diag++] = values[indx];
	                } 
	                else 
	 	        {
                           jx_error(JX_ERROR_GENERIC);
	    	           if (print_level)
                              jx_printf("Error in row %d ! Too many elements !\n", row);
	    	           /* return 1; */
                           return jx_error_flag;
	 	        }
	             } 
	             not_found = 1;
	          }
	          indx++;
	       }

               jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

            }
         }
         /* not my row */
         else
         {
   	    current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts = jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = jx_AuxParCSRMatrixOffProcData(aux_matrix);

	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = jx_max(n,1000);
	       jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
   	       jx_AuxParCSRMatrixOffProcI(aux_matrix) = jx_CTAlloc(JX_Int,2*max_off_proc_elmts);
   	       jx_AuxParCSRMatrixOffProcJ(aux_matrix) = jx_CTAlloc(JX_Int,max_off_proc_elmts);
   	       jx_AuxParCSRMatrixOffProcData(aux_matrix) = jx_CTAlloc(JX_Real,max_off_proc_elmts);
   	       off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = jx_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = jx_TReAlloc(off_proc_i,JX_Int,2*max_off_proc_elmts);
               off_proc_j = jx_TReAlloc(off_proc_j,JX_Int,max_off_proc_elmts);
               off_proc_data = jx_TReAlloc(off_proc_data,JX_Real,max_off_proc_elmts);
	       jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
	       jx_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       jx_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       jx_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = row; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    jx_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix) = current_num_elmts; 
         }
      }
   }

   return jx_error_flag;
}

JX_Int
JX_IJMatrixAddToValues( JX_IJMatrix   matrix,
                        JX_Int           nrows,
                        JX_Int          *ncols,
                        const JX_Int    *rows,
                        const JX_Int    *cols,
                        const JX_Real *values )
{
   jx_IJMatrix *ijmatrix = (jx_IJMatrix *) matrix;
   if (nrows == 0)
      return jx_error_flag;
   if (!ijmatrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (nrows < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   if (!ncols)
   {
      jx_error_in_arg(3);
      return jx_error_flag;
   }
   if (!rows)
   {
      jx_error_in_arg(4);
      return jx_error_flag;
   }
   if (!cols)
   {
      jx_error_in_arg(5);
      return jx_error_flag;
   }
   if (!values)
   {
      jx_error_in_arg(6);
      return jx_error_flag;
   }
   if (jx_IJMatrixObjectType(ijmatrix) == JX_PARCSR)
   {
      if (jx_IJMatrixOMPFlag(ijmatrix))
	 return( jx_IJMatrixAddToValuesOMPParCSR( ijmatrix, nrows, ncols, rows, cols, values ) );
      else
         return( jx_IJMatrixAddToValuesParCSR( ijmatrix, nrows, ncols, rows, cols, values ) );
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
JX_IJMatrixRead( const char  *filename,
                 MPI_Comm     comm,
                 JX_Int          type,
		 JX_IJMatrix *matrix_ptr )
{
   JX_IJMatrix matrix;
   JX_Int       ilower, iupper, jlower, jupper;
   JX_Int       ncols, I, J;
   JX_Real    value;
   JX_Int       myid, ret;
   char      new_filename[255];
   FILE     *file;

   jx_MPI_Comm_rank(comm, &myid);
   jx_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_fscanf(file, "%d %d %d %d", &ilower, &iupper, &jlower, &jupper);
   JX_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);
   JX_IJMatrixSetObjectType(matrix, type);
   JX_IJMatrixInitialize(matrix);

   /* It is important to ensure that whitespace follows the index value to help
    * catch mistakes in the input file.  See comments in IJVectorRead(). */
   ncols = 1;
   while ( (ret = jx_fscanf(file, "%d %d%*[ \t]%le", &I, &J, &value)) != EOF )
   {
      if (ret != 3)
      {
         jx_error_w_msg(JX_ERROR_GENERIC, "Error in IJ matrix input file.");
         return jx_error_flag;
      }
      if (I < ilower || I > iupper)
         JX_IJMatrixAddToValues(matrix, 1, &ncols, &I, &J, &value);
      else
         JX_IJMatrixSetValues(matrix, 1, &ncols, &I, &J, &value);
   }
   JX_IJMatrixAssemble(matrix);
   fclose(file);
  *matrix_ptr = matrix;

   return jx_error_flag;
}

JX_Int
JX_IJMatrixPrint( JX_IJMatrix matrix, const char *filename )
{
   MPI_Comm  comm;
   JX_Int      *row_partitioning;
   JX_Int      *col_partitioning;
   JX_Int       ilower, iupper, jlower, jupper;
   JX_Int       i, j, ii = 0;
   JX_Int       ncols, *cols;
   JX_Real   *values;
   JX_Int       myid;
   char      new_filename[255];
   FILE     *file;
   void     *object;

   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if ((jx_IJMatrixObjectType(matrix) != JX_PARCSR))
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   comm = jx_IJMatrixComm(matrix);
   jx_MPI_Comm_rank(comm, &myid);
   jx_sprintf(new_filename,"%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   col_partitioning = jx_IJMatrixColPartitioning(matrix);
#ifdef JX_NO_GLOBAL_PARTITION
   ilower = row_partitioning[0];
   iupper = row_partitioning[1] - 1;
   jlower = col_partitioning[0];
   jupper = col_partitioning[1] - 1;
#else
   ilower = row_partitioning[myid];
   iupper = row_partitioning[myid+1] - 1;
   jlower = col_partitioning[myid];
   jupper = col_partitioning[myid+1] - 1;
#endif
   jx_fprintf(file, "%d %d %d %d\n", ilower, iupper, jlower, jupper);
   JX_IJMatrixGetObject(matrix, &object);
   for (i = ilower; i <= iupper; i++)
   {
      if (jx_IJMatrixObjectType(matrix) == JX_PARCSR)
      {
#ifdef JX_NO_GLOBAL_PARTITION
         ii = i -  jx_IJMatrixGlobalFirstRow(matrix);
#else
         ii = i - row_partitioning[0];
#endif
         JX_ParCSRMatrixGetRow((JX_ParCSRMatrix) object, ii, &ncols, &cols, &values);
         for (j = 0; j < ncols; j++)
         {
#ifdef JX_NO_GLOBAL_PARTITION
            cols[j] +=  jx_IJMatrixGlobalFirstCol(matrix);
#else
            cols[j] += col_partitioning[0];
#endif
         }
      }
      for (j = 0; j < ncols; j++)
      {
         jx_fprintf(file, "%d %d %.14e\n", i, cols[j], values[j]);
      }
      if (jx_IJMatrixObjectType(matrix) == JX_PARCSR)
      {
         for (j = 0; j < ncols; j++)
         {
#ifdef JX_NO_GLOBAL_PARTITION
            cols[j] -=  jx_IJMatrixGlobalFirstCol(matrix);
#else
            cols[j] -= col_partitioning[0];
#endif
         }
         JX_ParCSRMatrixRestoreRow((JX_ParCSRMatrix) object, ii, &ncols, &cols, &values);
      }
   }
   fclose(file);

   return jx_error_flag;
}

JX_Int
jx_IJMatrixAddToValuesOMPParCSR( jx_IJMatrix  *matrix,
                                 JX_Int           nrows,
                                 JX_Int          *ncols,
                                 const JX_Int    *rows,
                                 const JX_Int    *cols,
                                 const JX_Real *values )
{
   jx_ParCSRMatrix *par_matrix;
   jx_CSRMatrix *diag, *offd;
   jx_AuxParCSRMatrix *aux_matrix;
   JX_Int *row_partitioning;
   JX_Int *col_partitioning;
   MPI_Comm comm = jx_IJMatrixComm(matrix);
   JX_Int num_procs, my_id;
   JX_Int col_0, col_n;
   JX_Int **aux_j = NULL;
   JX_Real **aux_data = NULL;
   JX_Int *row_length, *row_space;
   JX_Int need_aux;
   JX_Int first, pstart;
   JX_Int *diag_i = NULL;
   JX_Int *diag_j = NULL;
   JX_Real *diag_data = NULL;
   JX_Int *offd_i = NULL;
   JX_Int *offd_j = NULL;
   JX_Real *offd_data = NULL;
   JX_Int current_num_elmts;
   JX_Int max_off_proc_elmts;
   JX_Int off_proc_i_indx;
   JX_Int *off_proc_i;
   JX_Int *off_proc_j;
   JX_Real *off_proc_data;
   JX_Int *value_start, **offproc_cnt;
   JX_Int print_level = jx_IJMatrixPrintLevel(matrix);
   JX_Int max_num_threads;
   JX_Int error_flag = 0;
   JX_Int i1;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   max_num_threads = jx_NumThreads();
   par_matrix = jx_IJMatrixObject( matrix );
   row_partitioning = jx_IJMatrixRowPartitioning(matrix);
   col_partitioning = jx_IJMatrixColPartitioning(matrix);
   value_start = jx_CTAlloc(JX_Int, max_num_threads+1);
   offproc_cnt = jx_CTAlloc(JX_Int *, max_num_threads);
   for (i1=0; i1 < max_num_threads; i1++)
      offproc_cnt[i1] = NULL;

#ifdef JX_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first = jx_IJMatrixGlobalFirstCol(matrix);
   pstart = 0;
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
   pstart = my_id;
#endif
   if (jx_IJMatrixAssembleFlag(matrix)) /* matrix already assembled */
   {
      JX_Int num_cols_offd;
      JX_Int *col_map_offd = NULL;

      diag = jx_ParCSRMatrixDiag(par_matrix);
      diag_i = jx_CSRMatrixI(diag);
      diag_j = jx_CSRMatrixJ(diag);
      diag_data = jx_CSRMatrixData(diag);
      offd = jx_ParCSRMatrixOffd(par_matrix);
      offd_i = jx_CSRMatrixI(offd);
      num_cols_offd = jx_CSRMatrixNumCols(offd);
      if (num_cols_offd)
      {
          col_map_offd = jx_ParCSRMatrixColMapOffd(par_matrix);
          offd_j = jx_CSRMatrixJ(offd);
          offd_data = jx_CSRMatrixData(offd);
      }
      aux_matrix = jx_IJMatrixTranslator(matrix);
      if (aux_matrix)
      {
         current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
         off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
      }
#if JX_USING_OPENMP
#pragma omp parallel 
#endif
      {
         JX_Int j_offd;
         JX_Int num_threads, my_thread_num;
         JX_Int len, rest, ns, ne;
         JX_Int pos_diag, pos_offd;
         JX_Int len_diag, len_offd;
         JX_Int row_local;
         JX_Int i, j, ii, n, row;
         JX_Int not_found, size, indx;
         JX_Int *my_offproc_cnt = NULL;

         num_threads = jx_NumActiveThreads();
         my_thread_num = jx_GetThreadNum();

         len = nrows/num_threads; 
         rest = nrows - len*num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num*(len+1);
            ne = (my_thread_num+1)*(len+1);
         }
         else
         {
            ns = my_thread_num*len+rest;
            ne = (my_thread_num+1)*len+rest;
         }

         value_start[my_thread_num] = 0;
         for (ii=ns; ii < ne; ii++)
            value_start[my_thread_num] += ncols[ii];

#if JX_USING_OPENMP
#pragma omp barrier
#endif
         if (my_thread_num == 0)
         {
            for (i=0; i < max_num_threads; i++)
               value_start[i+1] += value_start[i];
         }
#if JX_USING_OPENMP
#pragma omp barrier
#endif
         indx = 0;
         if (my_thread_num) indx = value_start[my_thread_num-1];
         for (ii=ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols[ii];
            if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
            {
               row_local = row - row_partitioning[pstart]; 
               /* compute local row number */
               size = diag_i[row_local+1] - diag_i[row_local] + offd_i[row_local+1] - offd_i[row_local];
               if (n > size)
               {
                  jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                  error_flag++;
      	          if (print_level) jx_printf (" row %d too long! \n", row);
                  break;
               }
       
               pos_diag = diag_i[row_local];
               pos_offd = offd_i[row_local];
               len_diag = diag_i[row_local+1];
               len_offd = offd_i[row_local+1];
               not_found = 1;
      	
               for (i=0; i < n; i++)
               {
                  if (cols[indx] < col_0 || cols[indx] > col_n)
                  /* insert into offd */	
                  {
      	             j_offd = jx_BinarySearch(col_map_offd,cols[indx]-first,num_cols_offd);
      	             if (j_offd == -1)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
                        break;
      	             }
      	             for (j=pos_offd; j < len_offd; j++)
      	             {
      	                if (offd_j[j] == j_offd)
      	                {
                           offd_data[j] += values[indx];
      		           not_found = 0;
      		           break;
      	                }
      	             }
      	             if (not_found)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
                        break;
      	             }
      	             not_found = 1;
                  }
                  /* diagonal element */
      	          else if (cols[indx] == row)
      	          {
      	             if (diag_j[pos_diag] != row_local)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
                        break;
      	             }
      	             diag_data[pos_diag] += values[indx];
      	          }
                  else  /* insert into diag */
                  {
      	             for (j=pos_diag; j < len_diag; j++)
      	             {
      	                if (diag_j[j] == (cols[indx]-col_0))
      	                {
                           diag_data[j] += values[indx];
      		           not_found = 0;
      		           break;
      	                }
      	             }
      	             if (not_found)
      	             {
                        jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                        error_flag++;
      	                if (print_level)
			   jx_printf(" Error, element %d %d does not exist\n", row, cols[indx]);
                        break;
      	             }
                  }
                  indx++;
               }
            }
            /* not my row */
            /* need to find solution for threaded version!!!! */
            /* could save row number and process later .... */
            else
	    {
               if (!my_offproc_cnt)
               {
                  my_offproc_cnt = jx_CTAlloc(JX_Int, 200);
                  offproc_cnt[my_thread_num] = my_offproc_cnt;
                  my_offproc_cnt[0] = 200;
	          my_offproc_cnt[1] = 2;
               }
               i = my_offproc_cnt[1];
               if (i+2 < my_offproc_cnt[0])
               {
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i+1] = indx;
                  my_offproc_cnt[1] += 2;
               }
               else
               {
                  size = my_offproc_cnt[0];
                  my_offproc_cnt = jx_TReAlloc(my_offproc_cnt,JX_Int,size+200);
                  my_offproc_cnt[0] += 200;
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i+1] = indx;
                  my_offproc_cnt[1] += 2;
               }
               indx +=n;
	    }
         }
      } /* end parallel region */
   }
   
   /* not assembled */
   else
   {
      aux_matrix = jx_IJMatrixTranslator(matrix);
      if (aux_matrix)
      {
         current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
         off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
      }
      row_space = jx_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = jx_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = jx_AuxParCSRMatrixNeedAux(aux_matrix);
      if (need_aux)
      {
         aux_j = jx_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = jx_AuxParCSRMatrixAuxData(aux_matrix);
      }
      else
      {
         diag = jx_ParCSRMatrixDiag(par_matrix);
         diag_i = jx_CSRMatrixI(diag);
         diag_j = jx_CSRMatrixJ(diag);
         diag_data = jx_CSRMatrixData(diag);
         offd = jx_ParCSRMatrixOffd(par_matrix);
         offd_i = jx_CSRMatrixI(offd);
         if (num_procs > 1)
         {
            offd_j = jx_CSRMatrixJ(offd);
            offd_data = jx_CSRMatrixData(offd);
         }
      }
#if JX_USING_OPENMP
#pragma omp parallel 
#endif
      {
         JX_Int num_threads, my_thread_num;
         JX_Int len, rest, ns, ne;
         JX_Int *tmp_j = NULL;
         JX_Int *local_j = NULL;
         JX_Real *tmp_data = NULL;
         JX_Real *local_data = NULL;
         JX_Int tmp_indx;
         JX_Int row_local;
         JX_Int i, j, ii, n, row;
         JX_Int not_found, size, indx;
         JX_Int old_size, space, cnt;
         JX_Int *my_offproc_cnt = NULL;

         num_threads = jx_NumActiveThreads();
         my_thread_num = jx_GetThreadNum();

         len = nrows/num_threads; 
         rest = nrows - len*num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num*(len+1);
            ne = (my_thread_num+1)*(len+1);
         }
         else
         {
            ns = my_thread_num*len+rest;
            ne = (my_thread_num+1)*len+rest;
         }

         value_start[my_thread_num] = 0;
         for (ii=ns; ii < ne; ii++)
            value_start[my_thread_num] += ncols[ii];

#if JX_USING_OPENMP
#pragma omp barrier
#endif
         if (my_thread_num == 0)
         {
            for (i=0; i < max_num_threads; i++)
               value_start[i+1] += value_start[i];
         }
#if JX_USING_OPENMP
#pragma omp barrier
#endif
         indx = 0;
         if (my_thread_num) indx = value_start[my_thread_num-1];
         for (ii=ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols[ii];
            if (row >= row_partitioning[pstart] && row < row_partitioning[pstart+1])
            {
               row_local = row - row_partitioning[pstart]; 
               /* compute local row number */
               if (need_aux)
               {
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
   	          space = row_space[row_local]; 
   	          old_size = row_length[row_local]; 
   	          size = space - old_size;
   	          if (size < n)
      	          {
      	             size = n - size;
      	             tmp_j = jx_CTAlloc(JX_Int,size);
      	             tmp_data = jx_CTAlloc(JX_Real,size);
      	          }
      	          tmp_indx = 0;
      	          not_found = 1;
      	          size = old_size;
                  for (i=0; i < n; i++)
      	          {
      	             for (j=0; j < old_size; j++)
      	             {
      	                if (local_j[j] == cols[indx])
      	                {
                           local_data[j] += values[indx];
      		           not_found = 0;
      		           break;
      	                }
      	             }
      	             if (not_found)
      	             {
      	                if (size < space)
      	                {
      	                   local_j[size] = cols[indx];
      	                   local_data[size++] = values[indx];
      	                }
      	                else
      	                {
      	                   tmp_j[tmp_indx] = cols[indx];
      	                   tmp_data[tmp_indx++] = values[indx];
      	                }
      	             }
      	             not_found = 1;
        	     indx++;
      	          }
      	    
                  row_length[row_local] = size+tmp_indx;
               
                  if (tmp_indx)
                  {
   	             aux_j[row_local] = jx_TReAlloc(aux_j[row_local],JX_Int,size+tmp_indx);
   	             aux_data[row_local] = jx_TReAlloc(aux_data[row_local],JX_Real,size+tmp_indx);
                     row_space[row_local] = size+tmp_indx;
                     local_j = aux_j[row_local];
                     local_data = aux_data[row_local];
                  }
   
   	          cnt = size; 
   
   	          for (i=0; i < tmp_indx; i++)
   	          {
   	             local_j[cnt] = tmp_j[i];
   	             local_data[cnt++] = tmp_data[i];
	          }
  
	          if (tmp_j)
	          { 
	             jx_TFree(tmp_j); 
	             jx_TFree(tmp_data); 
	          } 
               }
               else /* insert immediately into data in ParCSRMatrix structure */
               {
                  JX_Int offd_indx, diag_indx;
                  JX_Int offd_space, diag_space;
                  JX_Int cnt_diag, cnt_offd;
	          offd_indx = jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	          diag_indx = jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
	          cnt_diag = diag_indx;
	          cnt_offd = offd_indx;
	          diag_space = diag_i[row_local+1];
	          offd_space = offd_i[row_local+1];
	          not_found = 1;
  	          for (i=0; i < n; i++)
	          {
	             if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */	
	             {
	                for (j=offd_i[row_local]; j < offd_indx; j++)
	                {
		           if (offd_j[j] == cols[indx])
		           {
                              offd_data[j] += values[indx];
		              not_found = 0;
		              break;
		           }
	                }
	                if (not_found)
	                { 
	                   if (cnt_offd < offd_space) 
	                   { 
	                      offd_j[cnt_offd] = cols[indx];
	                      offd_data[cnt_offd++] = values[indx];
	                   } 
	                   else 
	 	           {
                              jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                              error_flag++;
	    	              if (print_level)
                                 jx_printf("Error in row %d ! Too many elements!\n", row);
                              break;
	 	           }
	                }  
	                not_found = 1;
	             }
	             else  /* insert into diag */
	             {
	                for (j=diag_i[row_local]; j < diag_indx; j++)
	                {
		           if (diag_j[j] == cols[indx])
		           {
                              diag_data[j] += values[indx];
		              not_found = 0;
		              break;
		           }
	                } 
	                if (not_found)
	                { 
	                   if (cnt_diag < diag_space) 
	                   { 
	                      diag_j[cnt_diag] = cols[indx];
	                      diag_data[cnt_diag++] = values[indx];
	                   } 
	                   else 
	 	           {
                              jx_error(JX_ERROR_GENERIC);
#if JX_USING_OPENMP
#pragma omp atomic
#endif
                              error_flag++;
	    	              if (print_level)
                                 jx_printf("Error in row %d ! Too many elements !\n", row);
                              break;
	 	           }
	                } 
	                not_found = 1;
	             }
	             indx++;
	          }

                  jx_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
                  jx_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

               }
            }
            /* not my row */
            else
            {
//               JX_Int offproc_indx = 0;
               if (!my_offproc_cnt)
               {
                  my_offproc_cnt = jx_CTAlloc(JX_Int, 200);
                  offproc_cnt[my_thread_num] = my_offproc_cnt;
                  my_offproc_cnt[0] = 200;
	          my_offproc_cnt[1] = 2;
//	          offproc_indx = 2;
               }
               i = my_offproc_cnt[1];
               if (i+2 < my_offproc_cnt[0])
               {
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i+1] = indx;
                  my_offproc_cnt[1] += 2;
               }
               else
               {
                  size = my_offproc_cnt[0];
                  my_offproc_cnt = jx_TReAlloc(my_offproc_cnt,JX_Int,size+200);
                  my_offproc_cnt[0] += 200;
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i+1] = indx;
                  my_offproc_cnt[1] += 2;
               }
               indx +=n;
            }
         }
      } /*end parallel region */
   }
   if (error_flag) return jx_error_flag;
   jx_TFree(value_start);
   if (!aux_matrix)
   {
       JX_Int size = row_partitioning[pstart+1]-row_partitioning[pstart];
       jx_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
       jx_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
       jx_IJMatrixTranslator(matrix) = aux_matrix;
   }
   for (i1 = 0; i1 < max_num_threads; i1++)
   {
      if (offproc_cnt[i1])
      {
         JX_Int *my_offproc_cnt = offproc_cnt[i1];
         JX_Int i, i2, ii, row, n, indx;
         for (i2 = 2; i2 < my_offproc_cnt[1]; i2+=2)
         {
            ii = my_offproc_cnt[i2];
            row = rows[ii];
            n = ncols[ii];
            indx = my_offproc_cnt[i2+1];
   	    current_num_elmts = jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts = jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = jx_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = jx_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = jx_max(n,1000);
	       jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
   	       jx_AuxParCSRMatrixOffProcI(aux_matrix) = jx_CTAlloc(JX_Int,2*max_off_proc_elmts);
   	       jx_AuxParCSRMatrixOffProcJ(aux_matrix) = jx_CTAlloc(JX_Int,max_off_proc_elmts);
   	       jx_AuxParCSRMatrixOffProcData(aux_matrix) = jx_CTAlloc(JX_Real,max_off_proc_elmts);
   	       off_proc_i = jx_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = jx_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = jx_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = jx_TReAlloc(off_proc_i,JX_Int,2*max_off_proc_elmts);
               off_proc_j = jx_TReAlloc(off_proc_j,JX_Int,max_off_proc_elmts);
               off_proc_data = jx_TReAlloc(off_proc_data,JX_Real,max_off_proc_elmts);
	       jx_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
	       jx_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       jx_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       jx_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = row; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    jx_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    jx_AuxParCSRMatrixCurrentNumElmts(aux_matrix) = current_num_elmts; 
	 }
	 jx_TFree (offproc_cnt[i1]);
      }
   }
   jx_TFree(offproc_cnt);
   return jx_error_flag;
}

JX_Int
JX_IJMatrixGetObject( JX_IJMatrix matrix, void **object )
{
   jx_IJMatrix *ijmatrix = (jx_IJMatrix *) matrix;
   if (!ijmatrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
  *object = jx_IJMatrixObject(ijmatrix);

   return jx_error_flag;
}
