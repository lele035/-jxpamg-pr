//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_vector.c -- basic operations for sequential vectors
 *  Date: 2011/05/13
 */ 

#include "jx_mv.h"

/*!
 * \fn jx_Vector *jx_SeqVectorCreate
 * \brief Create a sequential vector with size of 'size'.
 * \note No memories for the vector are allocated here.
 * \date 2011/05/13
 */
jx_Vector *
jx_SeqVectorCreate( JX_Int size )
{
   jx_Vector *vector;

   vector = jx_CTAlloc(jx_Vector, 1);

   jx_VectorData(vector) = NULL;
   jx_VectorSize(vector) = size;

   jx_VectorNumVectors(vector) = 1;
   jx_VectorMultiVecStorageMethod(vector) = 0;

   /* set defaults */
   jx_VectorOwnsData(vector) = 1;

   return vector;
}

/*!
 * \fn JX_Int jx_SeqVectorInitialize
 * \brief Initialize a sequential vector.
 * \note Allocate memory for the vector if needed.
 * \date 2011/05/13
 */
JX_Int 
jx_SeqVectorInitialize( jx_Vector *vector )
{
   JX_Int  size                    = jx_VectorSize(vector);
   JX_Int  num_vectors             = jx_VectorNumVectors(vector);
   JX_Int  multivec_storage_method = jx_VectorMultiVecStorageMethod(vector);
   JX_Int  ierr                    = 0;   

   if ( !jx_VectorData(vector) )
   {
      jx_VectorData(vector) = jx_CTAlloc(JX_Real, num_vectors*size);
   }

   if ( multivec_storage_method == 0 )
   {
      jx_VectorVectorStride(vector) = size;
      jx_VectorIndexStride(vector)  = 1;
   }
   else if ( multivec_storage_method == 1 )
   {
      jx_VectorVectorStride(vector) = 1;
      jx_VectorIndexStride(vector)  = num_vectors;
   }
   else
   {
      ++ ierr;
   }
   
   return ierr;
}

/*!
 * \fn JX_Int jx_SeqVectorSetDataOwner
 * \brief Set the member data_owner of a sequential vector.
 * \date 2011/09/07
 */
JX_Int
jx_SeqVectorSetDataOwner( jx_Vector  *vector,
                          JX_Int         owns_data )
{
   JX_Int ierr = 0;

   jx_VectorOwnsData(vector) = owns_data;

   return ierr;
}

/*!
 * \fn JX_Int jx_SeqVectorDestroy
 * \brief Destroy a sequential vector.
 * \date 2011/05/13
 */ 
JX_Int 
jx_SeqVectorDestroy( jx_Vector *vector )
{
   JX_Int ierr = 0;

   if (vector)
   {
      if ( jx_VectorOwnsData(vector) )
      {
         jx_TFree(jx_VectorData(vector));
      }
      jx_TFree(vector);
   }

   return ierr;
}

/*!
 * \fn JX_Int jx_SeqVectorPrint
 * \brief Print a sequential vector into a given file.
 * \date 2011/05/13
 */  
JX_Int
jx_SeqVectorPrint( jx_Vector *vector, char *file_name )
{
   FILE    *fp = NULL;

   JX_Real  *data;
   JX_Int      size, num_vectors, vecstride, idxstride;
   
   JX_Int      i, j, ierr = 0;

   num_vectors = jx_VectorNumVectors(vector);
   vecstride   = jx_VectorVectorStride(vector);
   idxstride   = jx_VectorIndexStride(vector);
   data        = jx_VectorData(vector);
   size        = jx_VectorSize(vector);

   fp = fopen(file_name, "w");

   if (num_vectors == 1)
   {
      jx_fprintf(fp, "%d\n", size);
   }
   else
   {
      jx_fprintf(fp, "%d vectors of size %d\n", num_vectors, size);
   }

   if (num_vectors > 1)
   {
      for (j = 0; j < num_vectors; j ++)
      {
         jx_fprintf(fp, "vector %d\n", j);
         for (i = 0; i < size; i ++)
         {
            jx_fprintf(fp, "%.15e\n", data[j*vecstride + i*idxstride]);
         }
      }
   }
   else // if (num_vectors == 1)
   {
      for (i = 0; i < size; i ++)
      {
         jx_fprintf(fp, "%.15e\n", data[i]);
      }
   }

   fclose(fp);

   return ierr;
}

/*!
 * \fn jx_Vector *jx_SeqVectorRead
 * \brief Read a sequential vector from a given file.
 * \date 2011/05/13
 */ 
jx_Vector *
jx_SeqVectorRead( char *file_name )
{
   jx_Vector *vector;

   FILE      *fp;

   JX_Real    *data;
   JX_Int        size;
   
   JX_Int        j;

   fp = fopen(file_name, "r");

   jx_fscanf(fp, "%d", &size);

   vector = jx_SeqVectorCreate(size);
   jx_SeqVectorInitialize(vector);

   data = jx_VectorData(vector);
   for (j = 0; j < size; j ++)
   {
      jx_fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   /* multivector code not written yet */
   jx_assert( jx_VectorNumVectors(vector) == 1 );

   return vector;
}

jx_Vector *
jx_SeqVectorRead2( char *file_name )
{
   jx_Vector *vector;

   FILE      *fp;

   JX_Real    *data;
   JX_Int        size;
   
   JX_Int        j, tmp;

   fp = fopen(file_name, "r");

   jx_fscanf(fp, "%d", &size);

   vector = jx_SeqVectorCreate(size);
   jx_SeqVectorInitialize(vector);

   data = jx_VectorData(vector);
   for (j = 0; j < size; j ++)
   {
      jx_fscanf(fp, "%d%le", &tmp, &data[j]);
   }

   fclose(fp);

   /* multivector code not written yet */
   jx_assert( jx_VectorNumVectors(vector) == 1 );

   return vector;
}

/*!
 * \fn jx_Vector *jx_SeqMultiVectorCreate
 * \brief Create a multi vector.
 * \date 2011/05/13
 */
jx_Vector *
jx_SeqMultiVectorCreate( JX_Int size, JX_Int num_vectors )
{
   jx_Vector *vector = jx_SeqVectorCreate(size);
   jx_VectorNumVectors(vector) = num_vectors;
   return vector;
}
