//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

/*!
 * \fn JX_Int jx_OutOfMemory
 * \brief Out-of-memory warning.
 * \date 2011/09/01 
 */  
JX_Int
jx_OutOfMemory( size_t size )
{
   jx_printf(" Out of memory trying to allocate %d bytes\n", (JX_Int) size);
   fflush(stdout);

   jx_error(JX_ERROR_MEMORY);

   return 0;
}

/*!
 * \fn char *jx_MAlloc
 * \brief Allocating memory.
 * \date 2011/09/01 
 */  
char *
jx_MAlloc( size_t size )
{
   char *ptr;

   if (size > 0)
   {
      ptr = malloc(size);
      if (ptr == NULL)
      {
         jx_OutOfMemory(size);
      }
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*!
 * \fn char *jx_CAlloc
 * \brief Allocating memory.
 * \date 2011/09/01 
 */ 
char *
jx_CAlloc( size_t count, size_t elt_size )
{
   char  *ptr;
   size_t size = count*elt_size;  // Zhou Zhiyang & Yue Xiaoqiang 2012/11/03

   if (size > 0)
   {
      ptr = calloc(count, elt_size);
      if (ptr == NULL)
      {
         jx_OutOfMemory(size);
      }
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*!
 * \fn char *jx_ReAlloc
 * \brief Allocating memory.
 * \date 2011/09/01 
 */ 
char *
jx_ReAlloc( char *ptr, size_t size )
{
   if (ptr == NULL)
      ptr = malloc(size);
   else
      ptr = realloc(ptr, size);

   if ((ptr == NULL) && (size > 0))
   {
      jx_OutOfMemory(size);
   }

   return ptr;
}

/*!
 * \fn void jx_Free
 * \brief Free memory.
 * \date 2011/09/01 
 */ 
void
jx_Free( char *ptr )
{
   if (ptr) free(ptr);
}
