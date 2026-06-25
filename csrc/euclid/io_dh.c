//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  io_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#undef __FUNC__
#define __FUNC__ "jx_openFile_dh"
FILE *jx_openFile_dh( const char *filenameIN, const char *modeIN )
{
    JX_START_FUNC_DH
    FILE *fp = NULL;
    
    if ((fp = fopen(filenameIN, modeIN)) == NULL)
    {
        jx_sprintf(jx_msgBuf_dh, "can't open file: %s for mode %s\n", filenameIN, modeIN);
        JX_SET_ERROR(NULL, jx_msgBuf_dh);
    }
    JX_END_FUNC_VAL(fp)
}

#undef __FUNC__
#define __FUNC__ "jx_closeFile_dh"
void jx_closeFile_dh( FILE *fpIN )
{
    if (fclose(fpIN))
    {
        JX_SET_V_ERROR("attempt to close file failed");
    }
}

void jx_io_dh_print_ebin_mat_private( JX_Int m,
                                   JX_Int beg_row,
                                   JX_Int *rp,
                                   JX_Int *cval,
                                   JX_Real *aval,
                                   JX_Int *n2o,
                                   JX_Int *o2n,
                                   jx_Hash_i_dh hash,
                                   char *filename )
{
}

extern void jx_io_dh_read_ebin_mat_private( JX_Int *m, JX_Int **rp, JX_Int **cval, JX_Real **aval, char *filename )
{
}

void jx_io_dh_print_ebin_vec_private( JX_Int n,
                                   JX_Int beg_row,
                                   JX_Real *vals,
                                   JX_Int *n2o,
                                   JX_Int *o2n,
                                   jx_Hash_i_dh hash,
                                   char *filename )
{
}

void jx_io_dh_read_ebin_vec_private( JX_Int *n, JX_Real **vals, char *filename )
{
}
