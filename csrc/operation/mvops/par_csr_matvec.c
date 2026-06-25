//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_csr_matvec.c -- basic operations for parallel matrix-vector multiplication.
 *  Date: 2011/09/05
 */ 

#include "jx_mv.h"

JX_Real jx_total_elapsed_time_matvec = 0.0;

/*!
 * \fn JX_Int jx_ParCSRMatrixMatvec
 * \brief Perform y := alpha*A*x + beta*y.
 * \date 2011/09/05
 */
JX_Int
jx_ParCSRMatrixMatvec( JX_Real           alpha,
              	       jx_ParCSRMatrix *A,
                       jx_ParVector    *x,
                       JX_Real           beta,
                       jx_ParVector    *y  )
{
   jx_ParCSRCommHandle **comm_handle;
   jx_ParCSRCommPkg	*comm_pkg = jx_ParCSRMatrixCommPkg(A);
   jx_CSRMatrix         *diag     = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix         *offd     = jx_ParCSRMatrixOffd(A);
   jx_Vector            *x_local  = jx_ParVectorLocalVector(x);   
   jx_Vector            *y_local  = jx_ParVectorLocalVector(y);   
   JX_Int         num_rows = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int         num_cols = jx_ParCSRMatrixGlobalNumCols(A);

   jx_Vector *x_tmp;
   JX_Int        x_size = jx_ParVectorGlobalSize(x);
   JX_Int        y_size = jx_ParVectorGlobalSize(y);
   JX_Int        num_vectors   = jx_VectorNumVectors(x_local);
   JX_Int	      num_cols_offd = jx_CSRMatrixNumCols(offd);
   JX_Int        ierr = 0;
   JX_Int	      num_sends, i, j, jv, index, start;

   JX_Int        vecstride = jx_VectorVectorStride( x_local );
   JX_Int        idxstride = jx_VectorIndexStride( x_local );

   JX_Real     *x_tmp_data, **x_buf_data;
   JX_Real     *x_local_data = jx_VectorData(x_local);

   JX_Real      wall_time = 0.0;  /* for debugging instrumentation  */

   if (jx__global_mvcpu_flag) wall_time = jx_time_getWallclockSeconds();
 
  /*---------------------------------------------------------------------
   *  Check for size compatibility.  ParMatvec returns ierr = 11 if
   *  length of X doesn't equal the number of columns of A,
   *  ierr = 12 if the length of Y doesn't equal the number of rows
   *  of A, and ierr = 13 if both are true.
   *
   *  Because temporary vectors are often used in ParMatvec, none of 
   *  these conditions terminates processing, and the ierr flag
   *  is informational only.
   *--------------------------------------------------------------------*/
 
   jx_assert( idxstride > 0 );

    if (num_cols != x_size)
              ierr = 11;

    if (num_rows != y_size)
              ierr = 12;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 13;

    jx_assert( jx_VectorNumVectors(y_local)==num_vectors );

    if ( num_vectors == 1 )
    {
       x_tmp = jx_SeqVectorCreate( num_cols_offd );
    }
    else
    {
       jx_assert( num_vectors > 1 );
       x_tmp = jx_SeqMultiVectorCreate( num_cols_offd, num_vectors );
    }
    jx_SeqVectorInitialize(x_tmp);
    x_tmp_data = jx_VectorData(x_tmp);
   
    comm_handle = jx_CTAlloc(jx_ParCSRCommHandle*, num_vectors);

  /*---------------------------------------------------------------------
   * If there exists no CommPkg for A, a CommPkg is generated using
   * equally load balanced partitionings
   *--------------------------------------------------------------------*/
   
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(A);
      comm_pkg = jx_ParCSRMatrixCommPkg(A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   x_buf_data = jx_CTAlloc(JX_Real *, num_vectors);
   for (jv = 0; jv < num_vectors; ++ jv)
   {
      x_buf_data[jv] = jx_CTAlloc( JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) );
   }

   if ( num_vectors == 1 )
   {
      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
         {
            x_buf_data[0][index++] = x_local_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
   }
   else
   {
      for (jv = 0; jv < num_vectors; ++ jv)
      {
         index = 0;
         for (i = 0; i < num_sends; i ++)
         {
            start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
            {
                 x_buf_data[jv][index++] 
               = x_local_data[ jv*vecstride + idxstride*jx_ParCSRCommPkgSendMapElmt(comm_pkg,j) ];
            }
         }
      }
   }

   jx_assert( idxstride == 1 );
   
   /* >>> ... The assert is because the following loop only works for 'column' storage of a multivector <<<
      >>> This needs to be fixed to work more generally, at least for 'row' storage. <<<
      >>> This in turn, means either change CommPkg so num_sends is no.zones*no.vectors (not no.zones)
      >>> or, less dangerously, put a stride in the logic of CommHandleCreate (stride either from a
      >>> new arg or a new variable inside CommPkg).  Or put the num_vector iteration inside
      >>> CommHandleCreate (perhaps a new multivector variant of it). */
      
   for (jv = 0; jv < num_vectors; ++ jv)
   {
        comm_handle[jv] 
      = jx_ParCSRCommHandleCreate( 1, comm_pkg, x_buf_data[jv], &(x_tmp_data[jv*num_cols_offd]) );
   }

   jx_CSRMatrixMatvec(alpha, diag, x_local, beta, y_local);
   
   for (jv = 0; jv < num_vectors; ++ jv)
   {
      jx_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
   }
   jx_TFree(comm_handle);

   if (num_cols_offd) 
   {
      jx_CSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
   }

   jx_SeqVectorDestroy(x_tmp);
   x_tmp = NULL;
   for (jv = 0; jv < num_vectors; ++ jv) 
   {
      jx_TFree(x_buf_data[jv]);
   }
   jx_TFree(x_buf_data);

   if (jx__global_mvcpu_flag) jx_total_elapsed_time_matvec += (jx_time_getWallclockSeconds() - wall_time);

   return ierr;
}

/*!
 * \fn JX_Int jx_ParCSRMatrixMatvecT
 * \brief Perform y := alpha*A^T*x + beta*y.
 * \date 2011/09/05
 */
JX_Int
jx_ParCSRMatrixMatvecT( JX_Real           alpha,
                        jx_ParCSRMatrix *A,
                        jx_ParVector    *x,
                        JX_Real           beta,
                        jx_ParVector    *y  )
{
   jx_ParCSRCommHandle **comm_handle;
   jx_ParCSRCommPkg	*comm_pkg = jx_ParCSRMatrixCommPkg(A);
   jx_CSRMatrix         *diag     = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix         *offd     = jx_ParCSRMatrixOffd(A);
   jx_Vector            *x_local  = jx_ParVectorLocalVector(x);
   jx_Vector            *y_local  = jx_ParVectorLocalVector(y);
   jx_Vector            *y_tmp;
   
   JX_Int           vecstride = jx_VectorVectorStride( y_local );
   JX_Int           idxstride = jx_VectorIndexStride( y_local );
   JX_Real       *y_tmp_data, **y_buf_data;
   JX_Real       *y_local_data = jx_VectorData(y_local);

   JX_Int         num_rows = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int         num_cols = jx_ParCSRMatrixGlobalNumCols(A);
   JX_Int	       num_cols_offd = jx_CSRMatrixNumCols(offd);
   JX_Int         x_size = jx_ParVectorGlobalSize(x);
   JX_Int         y_size = jx_ParVectorGlobalSize(y);
   JX_Int         num_vectors = jx_VectorNumVectors(y_local);

   JX_Int         i, j, jv, index, start, num_sends;

   JX_Int         ierr = 0;

   JX_Real      wall_time = 0.0;  /* for debugging instrumentation  */

   if (jx__global_mvcpu_flag) wall_time = jx_time_getWallclockSeconds();

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;

    comm_handle = jx_CTAlloc(jx_ParCSRCommHandle *, num_vectors);

    if (num_vectors == 1)
    {
       y_tmp = jx_SeqVectorCreate(num_cols_offd);
    }
    else
    {
       y_tmp = jx_SeqMultiVectorCreate(num_cols_offd,num_vectors);
    }
    jx_SeqVectorInitialize(y_tmp);


  /*---------------------------------------------------------------------
   * If there exists no CommPkg for A, a CommPkg is generated using
   * equally load balanced partitionings
   *--------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(A);
      comm_pkg = jx_ParCSRMatrixCommPkg(A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   y_buf_data = jx_CTAlloc(JX_Real *, num_vectors);
   for (jv = 0; jv < num_vectors; ++ jv)
   {
      y_buf_data[jv] = jx_CTAlloc( JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) );
   }
   
   y_tmp_data = jx_VectorData(y_tmp);
   y_local_data = jx_VectorData(y_local);

   jx_assert( idxstride == 1 ); /* >>> only 'column' storage of multivectors implemented so far */

   if (num_cols_offd) 
   {
      // offdT is optional. Used only if it's present.
      if (A->offdT)
      {
         jx_CSRMatrixMatvec(alpha, A->offdT, x_local, 0.0, y_tmp);
      }
      else
      {
         jx_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);
      }
   }

   for (jv = 0; jv < num_vectors; ++ jv)
   {
      /* >>> this is where we assume multivectors are 'column' storage */
      comm_handle[jv] = jx_ParCSRCommHandleCreate( 2, comm_pkg, &(y_tmp_data[jv*num_cols_offd]), y_buf_data[jv] );
   }

   if (A->diagT)
   {
      // diagT is optional. Used only if it's present.
      jx_CSRMatrixMatvec(alpha, A->diagT, x_local, beta, y_local);
   }
   else
   {
      jx_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
   }

   for (jv = 0; jv < num_vectors; ++ jv)
   {
      jx_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
   }
   jx_TFree(comm_handle);

   if (num_vectors == 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
         {
            y_local_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)] += y_buf_data[0][index++];
         }
      }
   }
   else
   {
      for (jv = 0; jv < num_vectors; ++ jv)
      {
         index = 0;
         for (i = 0; i < num_sends; i ++)
         {
            start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
            {
                  y_local_data[ jv*vecstride + idxstride*jx_ParCSRCommPkgSendMapElmt(comm_pkg,j) ]
               += y_buf_data[jv][index++];
            }
         }
      }
   }
   
   jx_SeqVectorDestroy(y_tmp);
   y_tmp = NULL;
   for (jv = 0; jv < num_vectors; ++ jv) 
   {
      jx_TFree(y_buf_data[jv]);
   }
   jx_TFree(y_buf_data);

   if (jx__global_mvcpu_flag) jx_total_elapsed_time_matvec += (jx_time_getWallclockSeconds() - wall_time);

   return ierr;
}
