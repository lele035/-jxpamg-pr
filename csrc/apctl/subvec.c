//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  subvec.c -- subroutines to extract a or some sub-vectors from a 3tvector.
 *  Date: 2011/09/08
 *
 *  The 3T vectors are of the following form:
 *    /  \      /  \        
 *   | v1 |    | vr |  
 *   | v2 | or | ve |
 *   | v3 |    | vi |
 *    \  /      \  /   
 * 
 *  Created by peghoty
 *
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn JX_Int jx_3tGetSubVecs
 * \brief Get the three sub vectors of a 3t vector.
 * \author peghoty
 * \date 2012/01/16
 */
JX_Int
jx_3tGetSubVecs( jx_Vector  *f, 
                 jx_Vector **fR_ptr, 
                 jx_Vector **fE_ptr, 
                 jx_Vector **fI_ptr )
{
   jx_Vector *fR = NULL; 
   jx_Vector *fE = NULL; 
   jx_Vector *fI = NULL;

   JX_Real *f_data = jx_VectorData(f);
   JX_Real *fR_data = NULL;
   JX_Real *fE_data = NULL;
   JX_Real *fI_data = NULL;
   JX_Int size = jx_VectorSize(f);
   JX_Int n  = size / 3;
   JX_Int n2 = 2*n;
   JX_Int i;

   fR = jx_SeqVectorCreate(n);
   fE = jx_SeqVectorCreate(n);
   fI = jx_SeqVectorCreate(n);

   jx_SeqVectorInitialize(fR);
   jx_SeqVectorInitialize(fE);
   jx_SeqVectorInitialize(fI);

   fR_data = jx_VectorData(fR);
   fE_data = jx_VectorData(fE);
   fI_data = jx_VectorData(fI);
   
   for (i = 0; i < n; i ++)
   {
      fR_data[i] = f_data[i];
      fE_data[i] = f_data[i+n];
      fI_data[i] = f_data[i+n2];
   }

   *fR_ptr = fR;
   *fE_ptr = fE;
   *fI_ptr = fI;

   return (0);
}

JX_Int
jx_mgGetSubVecs( jx_Vector  *f,
                 jx_Vector **fR,
                 jx_Vector **fE_ptr,
                 jx_Vector **fI_ptr,
                 JX_Int      ng )
{
   jx_Vector *fE = NULL;
   jx_Vector *fI = NULL;

   JX_Real *f_data = jx_VectorData(f);
   JX_Real *fE_data = NULL;
   JX_Real *fI_data = NULL;
   JX_Int size = jx_VectorSize(f);
   JX_Int i,j,n,e_pos,i_pos;

   n = size / (ng + 2);
   e_pos = ng * n;
   i_pos = e_pos + n;

   for (j = 0; j < ng; j ++)
   {
      fR[j] = jx_SeqVectorCreate(n);
      jx_SeqVectorInitialize(fR[j]);
   }
   fE = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(fE);
   fI = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(fI);

   fE_data = jx_VectorData(fE);
   fI_data = jx_VectorData(fI);

   for (i = 0; i < n; i ++)
   {
      for (j = 0; j < ng; j ++)
      {
         jx_VectorData(fR[j])[i] = f_data[i+j*n];
      }
      fE_data[i] = f_data[i+e_pos];
      fI_data[i] = f_data[i+i_pos];
   }

  *fE_ptr = fE;
  *fI_ptr = fI;

   return (0);
}
