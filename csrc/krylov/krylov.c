//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  krylov.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"
#include "jx_krylov.h"

/*!
 * \fn char *jx_ParKrylovCAlloc
 */ 
char *
jx_ParKrylovCAlloc( JX_Int count, JX_Int elt_size )
{
   return ( jx_CAlloc( count, elt_size ) );
}

/*!
 * \fn JX_Int jx_ParKrylovFree
 */ 
JX_Int
jx_ParKrylovFree( char *ptr )
{
   JX_Int ierr = 0;
   jx_Free( ptr );
   return ierr;
}

/*!
 * \fn JX_Int jx_ParKrylovCommInfo
 */ 
JX_Int
jx_ParKrylovCommInfo( void *A, JX_Int *my_id, JX_Int *num_procs )
{
   MPI_Comm comm = jx_ParCSRMatrixComm ((jx_ParCSRMatrix *) A);
   jx_MPI_Comm_size(comm, num_procs);
   jx_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*!
 * \fn void *jx_ParKrylovCreateVector
 */ 
void *
jx_ParKrylovCreateVector( void *vvector )
{
   jx_ParVector *vector = vvector;
   jx_ParVector *new_vector;

   new_vector = jx_ParVectorCreate( jx_ParVectorComm(vector),
				    jx_ParVectorGlobalSize(vector),	
                                    jx_ParVectorPartitioning(vector) );
   jx_ParVectorSetPartitioningOwner(new_vector,0);
   jx_ParVectorInitialize(new_vector);

   return ( (void *) new_vector );
}

/*!
 * \fn void *jx_ParKrylovCreateVectorArray
 */ 
void *
jx_ParKrylovCreateVectorArray( JX_Int n, void *vvector )
{
   jx_ParVector  *vector = vvector;
   jx_ParVector **new_vector;
   JX_Int i, j;

   new_vector = jx_CTAlloc(jx_ParVector*, n);
   for (i = 0; i < n; i ++)
   {
      new_vector[i] = jx_ParVectorCreate( jx_ParVectorComm(vector),
				          jx_ParVectorGlobalSize(vector),	
                                          jx_ParVectorPartitioning(vector) );
      jx_ParVectorSetPartitioningOwner(new_vector[i], 0);
      if (jx_ParVectorInitialize(new_vector[i]) == JX_ERROR_MEMORY)  // Feng Chunsheng & Yue Xiaoqiang 2012/10/26
      {
         jx__kdim_memory_error = i - 1;
         jx_error_flag = 0;
         for (j = 0; j < i; j ++)
         {
            jx_ParVectorDestroy(new_vector[j]);
         }
         jx_TFree(new_vector);
         new_vector = NULL;
         break;
      }
   }

   return ( (void *) new_vector );
}

/*!
 * \fn JX_Int jx_ParKrylovDestroyVector
 */ 
JX_Int
jx_ParKrylovDestroyVector( void *vvector )
{
   jx_ParVector *vector = vvector;
   return( jx_ParVectorDestroy( vector ) );
}

/*!
 * \fn void *jx_ParKrylovMatvecCreate
 */ 
void *
jx_ParKrylovMatvecCreate( void *A, void *x )
{
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*!
 * \fn JX_Int jx_ParKrylovMatvec
 */
JX_Int
jx_ParKrylovMatvec( void   *matvec_data,
                    JX_Real  alpha,
                    void   *A,
                    void   *x,
                    JX_Real  beta,
                    void   *y   )
{
   return ( jx_ParCSRMatrixMatvec ( alpha,
                                    (jx_ParCSRMatrix *) A,
                                    (jx_ParVector *) x,
                                    beta,
                                    (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMatvecDestroy
 */
JX_Int
jx_ParKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*!
 * \fn JX_Real jx_ParKrylovInnerProd
 */
JX_Real
jx_ParKrylovInnerProd( void *x, void *y )
{
   return ( jx_ParVectorInnerProd( (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMassInnerProd
 */
JX_Int
jx_ParKrylovMassInnerProd( void *x, void **y, JX_Int k, JX_Int unroll, void  *result )
{
   return ( jx_ParVectorMassInnerProd( (jx_ParVector *) x, (jx_ParVector **) y, k, unroll, (JX_Real *)result ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMassDotpTwo
 */
JX_Int
jx_ParKrylovMassDotpTwo( void *x, void *y, void **z, JX_Int k, JX_Int unroll, void *result_x, void *result_y )
{
   return ( jx_ParVectorMassDotpTwo( (jx_ParVector *) x, (jx_ParVector *) y, (jx_ParVector **) z,
              k, unroll, (JX_Real *) result_x, (JX_Real *) result_y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovCopyVector
 */
JX_Int
jx_ParKrylovCopyVector( void *x, void *y )
{
   return ( jx_ParVectorCopy( (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovClearVector
 */
JX_Int
jx_ParKrylovClearVector( void *x )
{
   return ( jx_ParVectorSetConstantValues( (jx_ParVector *) x, 0.0 ) );
}

/*!
 * \fn JX_Int jx_ParKrylovScaleVector
 */
JX_Int
jx_ParKrylovScaleVector( JX_Real alpha, void *x )
{
   return ( jx_ParVectorScale( alpha, (jx_ParVector *) x ) );
}

/*!
 * \fn JX_Int jx_ParKrylovAxpy
 */
JX_Int
jx_ParKrylovAxpy( JX_Real alpha, void *x, void *y )
{
   return ( jx_ParVectorAxpy( alpha, (jx_ParVector *) x, (jx_ParVector *) y ) );
}

/*!
 * \fn JX_Int jx_ParKrylovMassAxpy
 */
JX_Int
jx_ParKrylovMassAxpy( JX_Real *alpha, void **x, void *y, JX_Int k, JX_Int unroll )
{
   return ( jx_ParVectorMassAxpy( alpha, (jx_ParVector **) x, (jx_ParVector *) y , k, unroll ) );
}

/*!
 * \fn JX_Int jx_ParKrylovIdentitySetup
 */
JX_Int
jx_ParKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JX_Int jx_ParKrylovIdentity
 */
JX_Int
jx_ParKrylovIdentity( void *vdata, void *A, void *b, void *x )
{
   return( jx_ParKrylovCopyVector( b, x ) );
}
