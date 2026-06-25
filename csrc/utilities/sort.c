//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

/*!
 * \fn void jx_swap
 * \brief Swap two components of an integer-array.
 * \param *v pointer to the array to be swapped. 
 * \param i one of the index to be swapped.
 * \param j the other index to be swapped.
 * \date 2011/09/01 
 */  
void 
jx_swap( JX_Int *v, JX_Int i, JX_Int j )
{
   JX_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*!
 * \fn void jx_swap1
 * \brief Swap two components of a JX_Real-array.
 * \param *v pointer to the array to be swapped. 
 * \param i one of the index to be swapped.
 * \param j the other index to be swapped.
 * \date 2011/09/01 
 */ 
void 
jx_swap1( JX_Real *v, JX_Int i, JX_Int j )
{
   JX_Real temp;
   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*!
 * \fn void jx_swap2
 * \brief Swap two components of an integer-array and a 
 *        JX_Real-array at the same time.
 * \param *v pointer to the integer-array to be swapped. 
 * \param *w pointer to the JX_Real-array to be swapped.  
 * \param i one of the index to be swapped.
 * \param j the other index to be swapped.
 * \date 2011/09/01 
 */   
void 
jx_swap2( JX_Int *v, JX_Real *w, JX_Int i, JX_Int j )
{
   JX_Int temp;
   JX_Real temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*!
 * \fn void jx_swap2i
 * \brief Swap two components of two integer-arrays at the same time.
 * \param *v pointer to the integer-array to be swapped. 
 * \param *w pointer to the other integer-array to be swapped.  
 * \param i one of the index to be swapped.
 * \param j the other index to be swapped.
 * \date 2011/09/01 
 */   
void 
jx_swap2i( JX_Int *v, JX_Int *w, JX_Int i, JX_Int j )
{
   JX_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*!
 * \fn void jx_swap3i
 * \brief Swap two components of three integer-arrays at the same time.
 * \param *v pointer to the first integer-array to be swapped. 
 * \param *w pointer to the second integer-array to be swapped. 
 * \param *z pointer to the third integer-array to be swapped.  
 * \param i one of the index to be swapped.
 * \param j the other index to be swapped.
 * \date 2011/09/01 
 */   
void 
jx_swap3i( JX_Int *v, JX_Int *w, JX_Int *z, JX_Int i, JX_Int j )
{
   JX_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/*!
 * \fn void jx_qsort0
 * \brief Sort parts components of an integer-array ascendingly.
 * \param *v pointer to the integer-array to be sorted. 
 * \param left starting index of the components to be sorted.
 * \param right ending index of the components to be sorted.
 * \date 2011/09/01 
 */   
void 
jx_qsort0( JX_Int *v, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right)
   {
      return;
   }
   jx_swap(v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i ++)
   {
      if (v[i] < v[left])
      {
         jx_swap(v, ++last, i);
      }
   }
   jx_swap(v, left, last);
   jx_qsort0(v, left, last-1);
   jx_qsort0(v, last+1, right);
}

/*!
 * \fn void jx_qsortd
 * \brief Sort parts components of a JX_Real-array ascendingly.
 * \param *v pointer to the JX_Real-array to be sorted. 
 * \param left starting index of the components to be sorted.
 * \param right ending index of the components to be sorted.
 * \date 2011/09/01 
 */ 
void 
jx_qsortd( JX_Real *v, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right)
   {
      return;
   }
   jx_swap1(v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i ++)
   {
      if (v[i] < v[left])
      {
         jx_swap1(v, ++last, i);
      }
   }
   jx_swap1(v, left, last);
   jx_qsortd(v, left, last-1);
   jx_qsortd(v, last+1, right);
}

/*!
 * \fn void jx_qsort1 (sort on w, move v)
 * \brief Sort the array 'v' ascendingly with quick sorting algorithm, 
 *        and meanwhile 'w' is sorted in the same way.
 * \param *v pointer to the JX_Int-array
 * \param *w pointer to the JX_Real-array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \date 2011/09/01  
 */  
void 
jx_qsort1( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right)
   {
      return;
   }
   jx_swap2(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i ++)
   {
      if (v[i] < v[left])
      {
         jx_swap2(v, w, ++last, i);
      }
   }   
   jx_swap2(v, w, left, last);
   jx_qsort1(v, w, left, last-1);
   jx_qsort1(v, w, last+1, right);
}

/*!
 * \fn void jx_qsort2i (sort on v, move w)
 * \brief Sort the array 'v' ascendingly with quick sorting algorithm, 
 *        and meanwhile 'w' is sorted in the same way.
 * \param *v pointer to the JX_Int-array
 * \param *w pointer to the JX_Int-array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \date 2011/09/01  
 */   
void 
jx_qsort2i( JX_Int *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right)
   {
      return;
   }
   jx_swap2i(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i ++)
   {
      if (v[i] < v[left])
      {
         jx_swap2i(v, w, ++last, i);
      }
   }
   jx_swap2i(v, w, left, last);
   jx_qsort2i(v, w, left, last-1);
   jx_qsort2i(v, w, last+1, right);
}

/*!
 * \fn void jx_qsort2 (sort on w, move v)
 * \brief Sort the array 'w' ascendingly with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int-array
 * \param *w pointer to the JX_Real-array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \date 2011/09/01  
 */  
void 
jx_qsort2( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right)
   {
      return;
   }
   jx_swap2(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i ++)
   {
      if (w[i] < w[left])
      {
         jx_swap2(v, w, ++last, i);
      }
   }
   jx_swap2(v, w, left, last);
   jx_qsort2(v, w, left, last-1);
   jx_qsort2(v, w, last+1, right);
}

/*!
 * \fn void jx_qsort3i (sort on v, move w and z)
 * \brief Sort the array 'v' ascendingly with quick sorting algorithm, 
 *        and meanwhile 'w', 'z' are sorted in the same way.
 * \param *v pointer to the first JX_Int-array
 * \param *w pointer to the second JX_Int-array
 * \param *z pointer to the third JX_Int-array 
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \date 2011/09/01  
 */  
void 
jx_qsort3i( JX_Int *v, JX_Int *w, JX_Int *z, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right)
   {
      return;
   }
   jx_swap3i(v, w, z, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i ++)
   {
      if (v[i] < v[left])
      {
         jx_swap3i(v, w, z, ++last, i);
      }
   }
   jx_swap3i(v, w, z, left, last);
   jx_qsort3i(v, w, z, left, last-1);
   jx_qsort3i(v, w, z, last+1, right);
}

/*!
 * \fn void jx_qsort2abs (sort on w, move v)
 * \brief Sort the array '|w|' ascendingly with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int-array
 * \param *w pointer to the JX_Real-array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \date 2011/09/01  
 */ 
void 
jx_qsort2abs( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i, last;
   if (left >= right)
   {
      return;
   }
   jx_swap2(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i ++)
   {
      if (fabs(w[i]) > fabs(w[left]))
      {
         jx_swap2(v, w, ++last, i);
      }
   }
   jx_swap2(v, w, left, last);
   jx_qsort2abs(v, w, left, last-1);
   jx_qsort2abs(v, w, last+1, right);
}

/*!
 * \fn void jx_diQuickSort12
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
jx_diQuickSort12( JX_Real *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmpw;
   JX_Real tmpv;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] <= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      jx_diQuickSort12(v, w, left, j-1);
      jx_diQuickSort12(v, w, i, right);
   }
}

/*!
 * \fn void jx_diQuickSort21
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
jx_diQuickSort21( JX_Real *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmpw;
   JX_Real tmpv;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] >= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      jx_diQuickSort21(v, w, left, j-1);
      jx_diQuickSort21(v, w, i, right);
   }
}
