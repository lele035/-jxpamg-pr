//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

JX_Int jx__global_mvcpu_flag = 0;

/*!
 * \fn void jx_set_mvcpu_handler
 * \brief Reset jx__global_mvcpu_flag.
 * \date 2018/01/16 
 */  
void
jx_set_mvcpu_handler( JX_Int new_mvcpu_flag )
{
   jx__global_mvcpu_flag = new_mvcpu_flag;
}
