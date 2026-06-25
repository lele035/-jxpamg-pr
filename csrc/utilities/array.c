//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

/*!
 * \fn void jx_DoubleArrayPrint
 * \brief Print an JX_Real array of size ndof into a given file.
 * \date 2011/09/08 
 */
JX_Int
jx_DoubleArrayPrint( JX_Real *x, JX_Int ndof, char *filename )
{
   FILE *fp = NULL;
   
   JX_Int i;
   
   fp = fopen(filename, "w");
   
   jx_fprintf(fp, "%d\n", ndof);   
   for (i = 0; i < ndof; i ++)
   {
      jx_fprintf(fp, "%.15le\n", x[i]);
   }
   
   fclose(fp); 
   return 0;
}

/*!
 * \fn JX_Int jx_IntegerArrayGetInterp
 * \brief Get an integer array for interpolation
 * \author Yue Xiaoqiang
 * \date 2012/10/12
 */
JX_Int
jx_IntegerArrayGetInterp( JX_Int *icor_interp, JX_Int n_fine, JX_Int nbl, JX_Int nbr )
{
   JX_Int ierr = 0;
   JX_Int length = 0;
   JX_Int myid, mybegin, myend, min, max;
   JX_Int num_threads;
   
   num_threads = jx_NumThreads();
#define JX_SMP_PRIVATE myid,mybegin,myend,min,max
#define JX_SMP_REDUCTION_OP +
#define JX_SMP_REDUCTION_VARS length
#include "../../include/jx_smp_forloop.h"
   for (myid = 0; myid < num_threads; myid ++)
   {
      JX_OMP_GET_START_END(myid, num_threads, n_fine, mybegin, myend);
      min = mybegin - 2 * nbl;
      if (min < 0)
      {
         min = 0;
      }
      icor_interp[2*myid] = min;
      max = myend + 2 * nbr;
      if (max > n_fine)
      {
         max = n_fine;
      }
      length = max - min + 1;
      icor_interp[2*myid+1] = length;
   }
   icor_interp[2*num_threads] = length;
   
   return ierr;
}

/*!
 * \fn JX_Int jx_IntegerArraySetConstantValues
 * \brief Set an integer array to be constant
 * \author Yue Xiaoqiang
 * \date 2012/10/13
 */
JX_Int
jx_IntegerArraySetConstantValues( JX_Int n, JX_Int *x, JX_Int val )
{
   JX_Int ierr = 0;
   JX_Int i;
   
#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < n; i ++) x[i] = val;
   
   return ierr;
}

/*!
 * \fn JX_Int jx_IntegerArrayModFine2Coarse
 * \brief Modify the fine_to_coarse in parallel to avoid the generation of coarse_to_fine
 * \author Yue Xiaoqiang
 * \date 2012/10/13
 */
JX_Int
jx_IntegerArrayModFine2Coarse( JX_Int num_rows_A, JX_Int *fine_to_coarse )
{
   JX_Int ierr = 0;
   JX_Int myid, mybegin, myend, i;
   JX_Int num_threads;
   
   num_threads = jx_NumThreads();
#define JX_SMP_PRIVATE myid,mybegin,myend,i
#include "../../include/jx_smp_forloop.h"
   for (myid = 0; myid < num_threads; myid ++)
   {
      JX_OMP_GET_START_END(myid, num_threads, num_rows_A, mybegin, myend);
      if (myid == 0)
      {
         mybegin ++;
      }
      for (i = mybegin; i < myend; i ++)
      {
         if (fine_to_coarse[i] < fine_to_coarse[i-1])
         {
            fine_to_coarse[i] = fine_to_coarse[i-1];
         }
      }
   }
   
   return ierr;
}

/*!
 * \fn JX_Int jx_IntegerArrayModFine2Coarse
 * \brief Avoid the generation of coarse_to_fine
 * \author Yue Xiaoqiang
 * \date 2012/10/13
 */
JX_Int
jx_IntegerArrayGenerateIcor( JX_Int num_rows_A,
                             JX_Int num_cols_P,
                             JX_Int *fine_to_coarse,
                             JX_Int nbl,
                             JX_Int nbr,
                             JX_Int *CF_marker,
                             JX_Int *icor )
{
   JX_Int ierr = 0;
   JX_Int lengthAA = 0, lengthPP = 0;
   JX_Int myid, FiveMyid, mybegin, myend, min_A, max_A;
   JX_Int i, first_f_node, min_P, max_P, myend_minus_one;
   JX_Int num_threads;
   
   num_threads = jx_NumThreads();
#define JX_SMP_PRIVATE myid,FiveMyid,mybegin,myend,min_A,max_A,i,first_f_node,min_P,max_P,myend_minus_one
#define JX_SMP_REDUCTION_OP +
#define JX_SMP_REDUCTION_VARS lengthAA,lengthPP
#include "../../include/jx_smp_forloop.h"
   for (myid = 0; myid < num_threads; myid ++)
   {
      FiveMyid = myid * 5;
      JX_OMP_GET_START_END(myid, num_threads, num_cols_P, mybegin, myend);
      icor[FiveMyid] = mybegin;
      
     /* Number of C Points less than the number of used threads, avoid to occur "Segment Error" */
      if (mybegin == myend)
      {
         lengthAA = 0;
         lengthPP = 0;
         icor[FiveMyid+1] = 0;
         icor[FiveMyid+3] = 0;
      }
      else
      {
        /* Get the coarse_to_fine BEGINS */
        /* min_A */
         first_f_node = jx_BinarySearch(fine_to_coarse, mybegin, num_rows_A);
        /* Here, i_start = first_f_node - 1, but when first_f_node = 0, may cause min_A be lager by 1 */
         for (i = first_f_node; i > -1; i --)
         {
            if (fine_to_coarse[i] != mybegin)
            {
               break;
            }
         }
         min_A = i + 1;
         min_A = jx_max(0, min_A-2*nbl);
        /* max_A */
         max_A = 0; /* Avoid the warning */
         myend_minus_one = myend - 1;
         first_f_node = jx_BinarySearch(fine_to_coarse, myend_minus_one, num_rows_A);
         for (i = first_f_node; i > -1; i --)
         {
            if (fine_to_coarse[i] != myend_minus_one)
            {
               max_A = i;
               break;
            }
         }
         max_A = jx_min(num_rows_A, max_A+2*nbr+1);
         lengthAA = max_A - min_A;
         icor[FiveMyid+1] = lengthAA;
        /* offset of A_marker_i */
         icor[FiveMyid+2] = min_A;
        /* Get the coarse_to_fine ENDS */
        /* min_P */
         min_P = 0; /* Avoid the warning */
         for (i = min_A; i >= 0; i --)
         {
            if (CF_marker[i] < 0 && CF_marker[i] != -3)
            {
               first_f_node = i;
               break;
            }
         }
         if (i != -1)
         {
            first_f_node -= nbl;
            if (first_f_node <= 0)
            {
               min_P = 0;
            }
            else
            {
               for (i = first_f_node; i >= 0; i --)
               {
                  if (CF_marker[i] >= 0)
                  {
                     min_P = fine_to_coarse[i];
                     break;
                  }
               }
               if (i == -1)
               {
                  min_P = 0;
               }
            }
         }
        /* max_P */
         max_P = 0; /* Avoid the warning */
         for (i = max_A-1; i < num_rows_A; i ++)
         {
            if (CF_marker[i] < 0 && CF_marker[i] != -3)
            {
               first_f_node = i;
               break;
            }
         }
         if (i == num_rows_A)
         {
            max_P = num_cols_P;
         }
         else
         {
            first_f_node += nbr;
            if (first_f_node >= num_rows_A)
            {
               max_P = num_cols_P;
            }
            else
            {
               for (i = first_f_node; i < num_rows_A; i ++)
               {
                  if (CF_marker[i] >= 0)
                  {
                     max_P = fine_to_coarse[i] + 1;
                     break;
                  }
               }
               if (i == num_rows_A)
               {
                  max_P = num_cols_P;
               }
            }
         }
         lengthPP = max_P - min_P;
         icor[FiveMyid+3] = lengthPP;
        /* offset of P_marker_i */
         icor[FiveMyid+4] = min_P;
      }
   }
  /* total length of {A_marker_i,i=0,...,num_threads-1} */
   icor[5*num_threads] = lengthAA;
  /* total length of {P_marker_i,i=0,...,num_threads-1} */
   icor[5*num_threads+1] = lengthPP;
   
   return ierr;
}

/*!
 * \fn JX_Int jx_IntegerArrayCopy
 * \brief Copy data from x to y.
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */ 
JX_Int
jx_IntegerArrayCopy( JX_Int size, JX_Int *x, JX_Int *y )
{
   JX_Int i, ierr = 0;
   
#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < size; i ++)
   {
      y[i] = x[i];
   }
   
   return ierr;
}

/*!
 * \fn JX_Int jx_DoubleArrayCopy
 * \brief Copy data from x to y.
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */ 
JX_Int
jx_DoubleArrayCopy( JX_Int size, JX_Real *x, JX_Real *y )
{
   JX_Int i, ierr = 0;
   
#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < size; i ++)
   {
      y[i] = x[i];
   }
   
   return ierr;
}

/*!
 * \fn JX_Int jx_DoubleArrayReciprocalMap
 * \brief Get the reciprocal of an array
 * \author Yue Xiaoqiang
 * \date 2014/03/18
 */
JX_Int
jx_DoubleArrayReciprocalMap( JX_Real *u, JX_Int n, JX_Int *map )
{
   JX_Int i, ierr = 0;
    
#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < n; i ++)
   {
      u[map[i]] = 1.0 / u[map[i]];
   }
   
   return ierr;
}

/*!
 * \fn jx_DyadicDoubleArrayCTAlloc
 * \author peghoty, Yue Xiaoqiang
 * \date 2011/10/26
 */
JX_Real **
jx_DyadicDoubleArrayCTAlloc( JX_Int first, JX_Int second )
{
    JX_Real **ptr = jx_CTAlloc(JX_Real *, first);
    JX_Int i;
    
    for (i = 0; i < first; i ++)
    {
        ptr[i] = jx_CTAlloc(JX_Real, second);
    }
    
    return ptr;
}

/*!
 * \fn jx_TeracidicDoubleArrayCTAlloc
 * \author Yue Xiaoqiang
 * \date 2014/11/03
 */
JX_Real ***
jx_TeracidicDoubleArrayCTAlloc( JX_Int first, JX_Int second, JX_Int third )
{
    JX_Real ***ptr = jx_CTAlloc(JX_Real **, first);
    JX_Int i, j;
    
    for (i = 0; i < first; i ++)
    {
        ptr[i] = jx_CTAlloc(JX_Real *, second);
        for (j = 0; j < second; j ++)
        {
            ptr[i][j] = jx_CTAlloc(JX_Real, third);
        }
    }
    
    return ptr;
}

/*!
 * \fn jx_TetravalentDoubleArrayCTAlloc
 * \author Yue Xiaoqiang
 * \date 2014/11/07
 */
JX_Real ****
jx_TetravalentDoubleArrayCTAlloc( JX_Int first, JX_Int second, JX_Int third, JX_Int fourth )
{
    JX_Real ****ptr = jx_CTAlloc(JX_Real ***, first);
    JX_Int i, j, k;
    
    for (i = 0; i < first; i ++)
    {
        ptr[i] = jx_CTAlloc(JX_Real **, second);
        for (j = 0; j < second; j ++)
        {
            ptr[i][j] = jx_CTAlloc(JX_Real *, third);
            for (k = 0; k < third; k ++)
            {
                ptr[i][j][k] = jx_CTAlloc(JX_Real, fourth);
            }
        }
    }
    
    return ptr;
}

/*!
 * \fn void jx_DyadicDoubleArrayFree
 * \author Yue Xiaoqiang
 * \date 2014/10/07
 */
void
jx_DyadicDoubleArrayFree( JX_Real **ptr, JX_Int first )
{
    JX_Int i;
    
    for (i = 0; i < first; i ++)
    {
        jx_Free((char *)ptr[i]);
    }
    jx_TFree(ptr);
}

/*!
 * \fn jx_TeracidicDoubleArrayFree
 * \author Yue Xiaoqiang
 * \date 2014/11/03
 */
void
jx_TeracidicDoubleArrayFree( JX_Real ***ptr, JX_Int first, JX_Int second )
{
    JX_Int i, j;
    
    for (i = 0; i < first; i ++)
    {
        for (j = 0; j < second; j ++)
        {
            jx_Free((char *)ptr[i][j]);
        }
        jx_Free((char *)ptr[i]);
    }
    jx_TFree(ptr);
}

/*!
 * \fn jx_TetravalentDoubleArrayFree
 * \author Yue Xiaoqiang
 * \date 2014/11/07
 */
void
jx_TetravalentDoubleArrayFree( JX_Real ****ptr, JX_Int first, JX_Int second, JX_Int third )
{
    JX_Int i, j, k;
    
    for (i = 0; i < first; i ++)
    {
        for (j = 0; j < second; j ++)
        {
            for (k = 0; k < third; k ++)
            {
                jx_Free((char *)ptr[i][j][k]);
            }
            jx_Free((char *)ptr[i][j]);
        }
        jx_Free((char *)ptr[i]);
    }
    jx_TFree(ptr);
}

/*!
 * \fn jx_DoubleArrayAbsMaxElement
 * \brief Find the abs. maximal value
 * \author Yue Xiaoqiang
 * \date 2014/10/24
 */
JX_Real
jx_DoubleArrayAbsMaxElement( JX_Real *x, JX_Int n )
{
    JX_Real elm = fabs(x[0]);
    JX_Int i;
    
    for (i = 1; i < n; i ++)
    {
        if (fabs(x[i]) > elm)
        {
            elm = fabs(x[i]);
        }
    }
    
    return elm;
}

/*!
 * \fn jx_DoubleArrayAbsMinElement
 * \brief Find the abs. minimal value
 * \author Yue Xiaoqiang
 * \date 2014/10/24
 */
JX_Real
jx_DoubleArrayAbsMinElement( JX_Real *x, JX_Int n )
{
    JX_Real elm = fabs(x[0]);
    JX_Int i;
    
    for (i = 1; i < n; i ++)
    {
        if (fabs(x[i]) < elm)
        {
            elm = fabs(x[i]);
        }
    }
    
    return elm;
}

/*!
 * \fn jx_DoubleArrayMaxElement
 * \brief Find the maximal value
 * \author Yue Xiaoqiang
 * \date 2014/10/24
 */
JX_Real
jx_DoubleArrayMaxElement( JX_Real *x, JX_Int n )
{
    JX_Real elm = x[0];
    JX_Int i;
    
    for (i = 1; i < n; i ++)
    {
        if (x[i] > elm)
        {
            elm = x[i];
        }
    }
    
    return elm;
}
