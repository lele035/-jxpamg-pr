//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

/*!
 * \fn JX_Int jx_BinarySearch
 * \brief performs a binary search for value on array list where 
 *        list needs to contain ordered nonnegative numbers.
 * \param *list pointer to the array to be searched. 
 * \param value the component to be searched.
 * \param list_length size of the array.
 * \note the routine returns the location of the value or -1
 * \date 2011/09/01 
 */ 
JX_Int 
jx_BinarySearch( JX_Int *list, JX_Int value, JX_Int list_length )
{
   JX_Int low, high, m;
   JX_Int not_found = 1;

   low = 0;
   high = list_length - 1; 
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (value < list[m])
      {
         high = m - 1;
      }
      else if (value > list[m])
      {
         low = m + 1;
      }
      else
      {
         not_found = 0;
         return m;
      }
   }
   return -1;
}
