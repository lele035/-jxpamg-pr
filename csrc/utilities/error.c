//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

#define JX_PRINT_ERRORS 1

/*!
 * \fn void jx_error_handler
 * \brief Process the error with code ierr raised in the given 
 *        line of the given source file.
 * \date 2011/09/01 
 */  
void 
jx_error_handler( char *filename, JX_Int line, JX_Int ierr, const char *msg )
{
   jx_error_flag |= ierr;

#ifdef JX_PRINT_ERRORS
   if (msg)
   {
      jx_fprintf(stderr, "jx-error in file \"%s\", line %d, error code = %d - %s\n", filename, line, ierr, msg);
   }
   else
   {
      jx_fprintf(stderr, "jx-error in file \"%s\", line %d, error code = %d\n", filename, line, ierr);
   }
#endif
}
