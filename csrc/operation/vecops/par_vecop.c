//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_vecop.c -- basic operations for parallel vectors
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_ParVectorCopy
 * \brief Copy x -> y.
 * \date 2011/09/03
 */ 
JX_Int
jx_ParVectorCopy( jx_ParVector *x, jx_ParVector *y )
{
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
   return jx_SeqVectorCopy(x_local, y_local);
}

/*!
 * \fn JX_Int jx_ParVectorSetConstantValues
 * \brief Set a parallel vector to be a given constant.
 * \date 2011/09/03
 */   
JX_Int
jx_ParVectorSetConstantValues( jx_ParVector *v, JX_Real value )
{
   jx_Vector *v_local = jx_ParVectorLocalVector(v);
           
   return jx_SeqVectorSetConstantValues(v_local,value);
}

/*!
 * \fn JX_Real jx_ParVectorNorm1
 * \brief Compute the 1-norm of vector x.
 * \author peghoty 
 * \date 2009/07/23
 */ 
JX_Real
jx_ParVectorNorm1( jx_ParVector *x )
{
   MPI_Comm      comm = jx_ParVectorComm(x);
   jx_Vector *x_local = jx_ParVectorLocalVector(x);

   JX_Real result = 0.0;
   JX_Real local_result = jx_SeqVectorNorm1(x_local);

   jx_MPI_Allreduce(&local_result, &result, 1, JX_MPI_REAL, MPI_MAX, comm);

   return result;
}

/*!
 * \fn JX_Real jx_ParVectorPointWiseRelNorm1
 * \brief Compute the pointwise relative 1-norm of vector x and y.
 * \author peghoty 
 * \date 2009/07/23
 */ 
JX_Real
jx_ParVectorPointWiseRelNorm1(jx_ParVector *x, jx_ParVector *y)
{
   MPI_Comm      comm = jx_ParVectorComm(x);
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
   
   JX_Real result = 0.0;
   JX_Real local_result = jx_SeqVectorPointWiseRelNorm1(x_local, y_local);
   
   jx_MPI_Allreduce(&local_result, &result, 1, JX_MPI_REAL, MPI_MAX, comm);

   return result;
}

/*!
 * \fn JX_Int jx_ParVectorScale
 * \brief Perform y := alpha*y. 
 * \date 2011/09/03
 */ 
JX_Int
jx_ParVectorScale( JX_Real alpha, jx_ParVector *y )
{
   jx_Vector *y_local = jx_ParVectorLocalVector(y);

   return jx_SeqVectorScale( alpha, y_local);
}

/*!
 * \fn JX_Int jx_ParVecMul
 * \brief Performs z = x*y, where z_i is defiend by x_i*y_i.
 * \note x, y, z should be allocated and have the same partitioning.
 * \author peghoty
 * \date 2011/09/09 
 */ 
JX_Int
jx_ParVecMul( jx_ParVector *x,
              jx_ParVector *y,
              jx_ParVector *z )
{
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
   jx_Vector *z_local = jx_ParVectorLocalVector(z);
   
   jx_SeqVecMul(x_local, y_local, z_local);
   
   return 0;
}

/*!
 * \fn JX_Int jx_ParVecZXY
 * \brief Performs z = z + alpha*x*y, where xy_i is defiend by x_i*y_i.
 * \note x, y, z should be allocated and have the same partitioning.
 * \author peghoty
 * \date 2011/09/09 
 */ 
JX_Int
jx_ParVecZXY( jx_ParVector  *z,
              JX_Real         alpha,
              jx_ParVector  *x,
              jx_ParVector  *y )
{
   jx_Vector *z_local = jx_ParVectorLocalVector(z);
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
     
   if (alpha == 0) 
   {
      return 0;
   }
     
   jx_SeqVecZXY(z_local, alpha, x_local, y_local);
   
   return 0;
}

/*!
 * \fn JX_Real jx_ParVectorInnerProd
 * \brief Compute the inner product of two parallel vectors.
 * \date 2011/09/03
 */ 
JX_Real
jx_ParVectorInnerProd( jx_ParVector *x, jx_ParVector *y )
{
   MPI_Comm      comm = jx_ParVectorComm(x);
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
           
   JX_Real result = 0.0;
   JX_Real local_result = jx_SeqVectorInnerProd(x_local, y_local);
   
   jx_MPI_Allreduce(&local_result, &result, 1, JX_MPI_REAL, MPI_SUM, comm);
   
   return result;
}

JX_Int
jx_ParVectorMassInnerProd( jx_ParVector *x, jx_ParVector **y, JX_Int k, JX_Int unroll, JX_Real *result )
{
   MPI_Comm   comm    = jx_ParVectorComm(x);
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   JX_Real *local_result;
   JX_Int i;
   jx_Vector **y_local;

   y_local = jx_TAlloc(jx_Vector *, k);
   for (i = 0; i < k; i ++)
   {
      y_local[i] = (jx_Vector *)jx_ParVectorLocalVector(y[i]);
   }
   local_result = jx_CTAlloc(JX_Real, k);
   jx_SeqVectorMassInnerProd(x_local, y_local, k, unroll, local_result);
   jx_MPI_Allreduce(local_result, result, k, JX_MPI_REAL, MPI_SUM, comm);
   jx_TFree(y_local);
   jx_TFree(local_result);
 
   return jx_error_flag;
}

/*!
 * \fn JX_Real jx_ParVectorNorm2
 * \brief Compute the 2-norm of vector x.
 * \author peghoty 
 * \date 2009/07/23
 */ 
JX_Real
jx_ParVectorNorm2( jx_ParVector *x )
{
   JX_Real result = 0.0;
   result = sqrt( jx_ParVectorInnerProd(x,x) );
   return result;
}

/*!
 * \fn JX_Int jx_ParVectorAxpy
 * \brief Perform y = alpha*x + y 
 * \date 2011/09/03
 */ 
JX_Int
jx_ParVectorAxpy( JX_Real        alpha,
                  jx_ParVector *x,
                  jx_ParVector *y  )
{
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
   
   return jx_SeqVectorAxpy( alpha, x_local, y_local);
}

JX_Int
jx_ParVectorMassDotpTwo( jx_ParVector *x, jx_ParVector *y, jx_ParVector **z, JX_Int k, JX_Int unroll, JX_Real *result_x, JX_Real *result_y )
{
   MPI_Comm   comm    = jx_ParVectorComm(x);
   jx_Vector *x_local = jx_ParVectorLocalVector(x);
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
   JX_Real *local_result, *result;
   JX_Int i; 
   jx_Vector **z_local;
   z_local = jx_TAlloc(jx_Vector *, k);

   for (i = 0; i < k; i ++)
      z_local[i] = (jx_Vector *) jx_ParVectorLocalVector(z[i]);

   local_result = jx_CTAlloc(JX_Real, 2*k);
   result = jx_CTAlloc(JX_Real, 2*k);

   jx_SeqVectorMassDotpTwo(x_local, y_local, z_local, k, unroll, &local_result[0], &local_result[k]);

   jx_MPI_Allreduce(local_result, result, 2*k, JX_MPI_REAL, MPI_SUM, comm);

   for (i=0; i < k; i++)
   {
      result_x[i] = result[i];
      result_y[i] = result[k+i];
   }
   jx_TFree(z_local);
   jx_TFree(local_result);
   jx_TFree(result);
 
   return jx_error_flag;
}

JX_Int
jx_ParVectorMassAxpy( JX_Real *alpha, jx_ParVector **x, jx_ParVector *y, JX_Int k, JX_Int unroll )
{
   JX_Int i;
   jx_Vector **x_local;
   jx_Vector *y_local = jx_ParVectorLocalVector(y);
   x_local = jx_TAlloc(jx_Vector *, k);

   for (i=0; i < k; i++)
      x_local[i] = jx_ParVectorLocalVector(x[i]);

   jx_SeqVectorMassAxpy(alpha, x_local, y_local, k, unroll);

   jx_TFree(x_local);

   return jx_error_flag;
}
