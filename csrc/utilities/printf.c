//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"
#include <stdarg.h>
#include <stdio.h>

/* these prototypes are missing by default for some compilers */
int vscanf( const char *format , va_list arg );
int vfscanf( FILE *stream , const char *format, va_list arg );
int vsscanf( const char *s , const char *format, va_list arg );

JX_Int
jx_new_format( const char *format, char **newformat_ptr )
{
   const char *fp;
   char       *newformat, *nfp;
   JX_Int   newformatlen;
   JX_Int   foundpercent = 0;

   newformatlen = 2*strlen(format)+1; /* worst case is all %d's to %lld's */
   newformat = jx_TAlloc(char, newformatlen);

   nfp = newformat;
   for (fp = format; *fp != '\0'; fp ++)
   {
      if (*fp == '%')
      {
         foundpercent = 1;
      }
      else if (foundpercent)
      {
         if (*fp == 'l')
         {
            fp ++; /* remove 'l' and maybe add it back in switch statement */
            if (*fp == 'l')
            {
               fp ++; /* remove second 'l' if present */
            }
         }
         switch (*fp)
         {
            case 'd':
            case 'i':
#if JX_USING_BIG_INT
              *nfp = 'l'; nfp ++;
              *nfp = 'l'; nfp ++;
#endif
               foundpercent = 0; break;
            case 'f':
            case 'e':
            case 'E':
            case 'g':
            case 'G':
#if JX_USING_BIG_DOUBLE
              *nfp = 'L'; nfp++; /* modify with 'L' */
#else
              *nfp = 'l'; nfp++; /* modify with 'l' (default is double) */
#endif
               foundpercent = 0; break;
            case 'c':
            case 'n':
            case 'o':
            case 'p':
            case 's':
            case 'u':
            case 'x':
            case 'X':
            case '%':
               foundpercent = 0; break;
         }
      }
     *nfp = *fp; nfp ++;
   }
  *nfp = *fp; nfp ++;

  *newformat_ptr = newformat;

   return 0;
}

JX_Int
jx_free_format( char *newformat )
{
   jx_TFree(newformat);
   return 0;
}

/* printf functions */

JX_Int
jx_printf( const char *format, ... )
{
   va_list   ap;
   char     *newformat;
   JX_Int ierr = 0;

   va_start(ap, format);
   jx_new_format(format, &newformat);
   ierr = vprintf(newformat, ap);
   jx_free_format(newformat);
   va_end(ap);

   return ierr;
}

JX_Int
jx_fprintf( FILE *stream, const char *format, ... )
{
   va_list   ap;
   char     *newformat;
   JX_Int ierr = 0;

   va_start(ap, format);
   jx_new_format(format, &newformat);
   ierr = vfprintf(stream, newformat, ap);
   jx_free_format(newformat);
   va_end(ap);

   return ierr;
}

JX_Int
jx_sprintf( char *s, const char *format, ... )
{
   va_list   ap;
   char     *newformat;
   JX_Int ierr = 0;

   va_start(ap, format);
   jx_new_format(format, &newformat);
   ierr = vsprintf(s, newformat, ap);
   jx_free_format(newformat);
   va_end(ap);

   return ierr;
}

/* scanf functions */

JX_Int
jx_scanf( const char *format, ... )
{
   va_list   ap;
   char     *newformat;
   JX_Int ierr = 0;

   va_start(ap, format);
   jx_new_format(format, &newformat);
   ierr = vscanf(newformat, ap);
   jx_free_format(newformat);
   va_end(ap);

   return ierr;
}

JX_Int
jx_fscanf( FILE *stream, const char *format, ... )
{
   va_list   ap;
   char     *newformat;
   JX_Int ierr = 0;

   va_start(ap, format);
   jx_new_format(format, &newformat);
   ierr = vfscanf(stream, newformat, ap);
   jx_free_format(newformat);
   va_end(ap);

   return ierr;
}

JX_Int
jx_sscanf( char *s, const char *format, ... )
{
   va_list   ap;
   char     *newformat;
   JX_Int ierr = 0;

   va_start(ap, format);
   jx_new_format(format, &newformat);
   ierr = vsscanf(s, newformat, ap);
   jx_free_format(newformat);
   va_end(ap);

   return ierr;
}
