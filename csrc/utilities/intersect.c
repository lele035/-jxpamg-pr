//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

/*!
 * \fn JX_Int jx_MultiSetsIntersection
 * \brief Find the intersection of multi integer-sets.
 * \param numset the number of sets 
 * \param **set set[i][j] (i=0(1)numset-1, j=0(1)size[i]-1) stores entries of the i-th Set.
 * \param *size size[i] (i=0(1)numset-1) is size of the i-th Set. 
 * \param **c_ptr pointer to pointer to the intersection array.
 * \param *nc_ptr pointer to the length of the intersection array. 
 * \param *empty_ptr pointer to a mark to indicate whether all the sets have intersection.
 * \author peghoty
 * \date 2009/08/17 
 */  
JX_Int
jx_MultiSetsIntersection( JX_Int   numset, 
                          JX_Int **set, 
                          JX_Int  *size, 
                          JX_Int **c_ptr, 
                          JX_Int  *nc_ptr, 
                          JX_Int  *empty_ptr )
{
   JX_Int  i;
   JX_Int *c = NULL;
   JX_Int  nc = 0;
   JX_Int  empty;
   JX_Int  print_level = 0;
   JX_Int *tmpset = NULL;
   JX_Int  tmpsize = 0;

   tmpset  = set[0];
   tmpsize = size[0]; 
   
   for (i = 0; i < numset-1; i ++)
   {
      jx_TwoSetsIntersection(tmpset, tmpsize, set[i+1], size[i+1], &c, &nc, &empty);
     
      if (empty == 1)
      {
         break;
      }
      else
      {
         if (i > 0 && tmpset) free(tmpset);  // 释放比较过程中产生的交集
         tmpset = c;
         tmpsize = nc;
      }
   }
   
   if (print_level)
   {
      if (empty)
      {
         jx_printf("There doesn't exist intersection!\n\n");
      }
      else
      {
         for (i = 0; i < nc; i ++) jx_printf("c[%d] = %d\n", i, c[i]);
      } 
   }
   
   *c_ptr = c;
   *nc_ptr = nc;
   *empty_ptr = empty;
   
   return(0);
}

/*!
 * \fn JX_Int jx_TwoSetsIntersection
 * \brief Find the intersection of two integer-sets.
 * \param *a pointer to the first JX_Int-type array
 * \param na the length of "a" array
 * \param *b pointer to the second JX_Int-type array
 * \param nb the length of "b" array  
 * \param **c_ptr pointer to pointer to the intersection array.
 * \param *nc_ptr pointer to the length of the intersection array. 
 * \param *empty_ptr pointer to a mark to indicate whether the two sets have intersection.
 * \note  All the elements in "a" or "b" should be different!
 * \author peghoty
 * \date 2009/08/17 
 */ 
JX_Int
jx_TwoSetsIntersection( JX_Int  *a, 
                        JX_Int   na, 
                        JX_Int  *b, 
                        JX_Int   nb,
                        JX_Int **c_ptr, 
                        JX_Int  *nc_ptr, 
                        JX_Int  *empty_ptr )
{
   JX_Int  i, j;
   JX_Int  nc;
   JX_Int *c = NULL;
   JX_Int *d;            // 辅助数组
   JX_Int  empty = 0;    // 标志变量
   JX_Int  amin, amax;
   JX_Int  bmin, bmax;
   JX_Int  left, right;
   JX_Int  size; 
   JX_Int  print_level = 0;  // 打印开关 
   
   /* 找数组 a 中的最大最小值 */
   amax = a[0];
   amin = a[0];
   for (i = 1; i < na; i ++)
   {
      if (a[i] > amax) amax = a[i];
      if (a[i] < amin) amin = a[i];
   }
   
   if (print_level)
   {
     jx_printf("\namax = %d\n", amax);
     jx_printf("amin = %d\n\n", amin);
   }
   
   /* 找数组 b 中的最大最小值 */
   bmax = b[0];
   bmin = b[0];
   for (i = 1; i < nb; i ++)
   {
      if (b[i] > bmax) bmax = b[i];
      if (b[i] < bmin) bmin = b[i];
   }
 
    if (print_level)
   {
     jx_printf("bmax = %d\n", bmax);
     jx_printf("bmin = %d\n\n", bmin);
   }
 
   if (amax < bmin || amin > bmax)
   {  // 交集为空
      nc = 0;
      empty = 1;
      if (print_level)
      jx_printf("Case 1: There doesn't exist intersection between a and b!\n\n");
   }
   else if (amax == bmin || amin == bmax) 
   {  // 交集只有一个元素
      nc = 1;
      c = (JX_Int *)malloc(nc*sizeof(JX_Int));
      if (amax == bmin)
        c[0] = amax;
      else
        c[0] = amin;
   } 
   else
   {  // 交集可能有多个元素
      left = amin;
      if (bmin < left) left = bmin;
      right = amax;
      if (bmax > right) right = bmax;

      size = right - left + 1;
 
      if (print_level)
      {
        jx_printf("left = %d\n", left);
        jx_printf("right = %d\n\n", right);
        jx_printf("size = %d\n\n", size);
      }
      
      d = (JX_Int *)malloc(size*sizeof(JX_Int));
      for (i = 0; i < size; i ++) d[i] = 0;      // 初始化
       
      nc = 0;  // 初始化, nc 用来统计数组 c 的长度
      /*--------------------------------------------------------
       * Remark: 下面生成辅助数组 d 时，同时统计 nc，对于数组
       * a，b 中元素值间距比较大的情形，这样做，与生成数组 d 后
       * 再根据条件 d[j] > 1 来统计 nc 相比，可以减少判断的次
       * 数（前者只要 na+nb 次，后者需要 size 次），从而提高效率.
      ---------------------------------------------------------*/
      for (i = 0; i < na; i ++)
      {
        j = a[i]-left;
        d[j] ++;
        if (d[j] > 1) nc ++;
      }
      for (i = 0; i < nb; i ++)
      {
        j = b[i] - left;
        d[j] ++;
        if (d[j] > 1) nc ++;
      }
      
      if (print_level) jx_printf("nc = %d\n\n", nc);
      
      if (nc > 0) 
      {
         /* 生成交集 c, c 中元素按升序排列 */
         c = (JX_Int *)malloc(nc*sizeof(JX_Int));
         nc = 0;
         for (i = 0; i < size; i ++)
         {
           if (d[i] > 1) c[nc++] = i + left;
         }
      }
      else
      {
         empty = 1;
         if (print_level)
         jx_printf("Case 3: There doesn't exist intersection between a and b!\n\n");
      }
      free(d);
   }

   if (print_level) 
   {
      for (i = 0; i < nc; i ++)
      {
        jx_printf("c[%d] = %d\n", i, c[i]);
      }
   }

   *nc_ptr = nc;
   *c_ptr = c;
   *empty_ptr = empty;
      
   return(0);
}
