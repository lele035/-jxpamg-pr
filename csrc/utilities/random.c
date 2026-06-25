//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

/* Static variables */
static JX_Int Seed = 13579;

#define a  16807
#define m  2147483647
#define q  127773
#define r  2836

/*!
 * \fn void jx_SeedRand
 * \brief Initializes the pseudo-random number generator 
 *        to a place in the sequence.
 * \param seed an JX_Int containing the seed for the RNG.
 * \date 2011/09/01 
 */ 
void  
jx_SeedRand( JX_Int seed )
{
   Seed = seed;
}

/*!
 * \fn JX_Real jx_Rand
 * \brief Computes the next pseudo-random number in the sequence 
 *        using the global variable Seed.
 * \return a JX_Real containing the next number in the sequence divided by
 *         2147483647 so that the numbers are in (0, 1].
 * \date 2011/09/01 
 */
JX_Real 
jx_Rand()
{
   JX_Int  low, high, test;

   high = Seed / q;
   low  = Seed % q;
   test = a * low - r * high;
   if(test > 0)
   {
      Seed = test;
   }
   else
   {
      Seed = test + m;
   }

   return ((JX_Real)(Seed) / m);
}
