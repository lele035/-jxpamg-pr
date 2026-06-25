/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 * smoother.c  
 *
 * Created by peghoty 2010/08/30
 * Xiangtan University
 * peghoty@163.com
 *  
 */

#include "jx_multils.h"

/*!
 * \fn fsls_AMGRelaxDirect
 * \brief Gaussian Elimination solver
 * \param *A pointer to the matrix to be solved
 * \param *f pointer to the rhs vector
 * \param *u pointer to the approxiation
 * \author peghoty
 * \date 2010/08/30 
 */
JX_Int  
fsls_AMGRelaxGE( JX_Real         *full,
                 fsls_CSRMatrix *A,
                 fsls_Vector    *f,
                 fsls_Vector    *u )
{
   JX_Int     n      = fsls_CSRMatrixNumRows(A);
   JX_Real *A_data = fsls_CSRMatrixData(A);
   JX_Int    *A_i    = fsls_CSRMatrixI(A);
   JX_Int    *A_j    = fsls_CSRMatrixJ(A);
   JX_Real *u_data = fsls_VectorData(u);
   JX_Real *f_data = fsls_VectorData(f);            

   JX_Int i,jj;
   JX_Int column;
   JX_Int error_flag = 0;

   /* zero-out the 'full' array */
   memset(full, 0X0, n*n*sizeof(JX_Real));

   /* data preparing */
   for (i = 0; i < n; i ++)
   {
      for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
      {
         column = A_j[jj];
         full[i*n+column] = A_data[jj];
      }
      u_data[i] = f_data[i];
   }

   /* Direct solve: use gaussian elimination */
   error_flag = fsls_gselim(full, u_data, n);
  
   return (error_flag);
}

/*!
 * \fn fsls_AMGRelaxDirect
 * \brief Gaussian Elimination solver
 * \param *A pointer to the matrix to be solved
 * \param *f pointer to the rhs vector
 * \param *u pointer to the approxiation
 * \author peghoty
 * \date 2010/08/30 
 */
JX_Int  
fsls_AMGRelaxGEP( JX_Real         *full,
                  fsls_CSRMatrix *A,
                  fsls_Vector    *f,
                  fsls_Vector    *u )
{
   JX_Int     n      = fsls_CSRMatrixNumRows(A);
   JX_Real *A_data = fsls_CSRMatrixData(A);
   JX_Int    *A_i    = fsls_CSRMatrixI(A);
   JX_Int    *A_j    = fsls_CSRMatrixJ(A);
   JX_Real *u_data = fsls_VectorData(u);
   JX_Real *f_data = fsls_VectorData(f);
   
   JX_Int i,jj;
   JX_Int column;
   JX_Int error_flag = 0;

   /* zero-out the 'full' array */
   memset(full, 0X0, n*n*sizeof(JX_Real));

   /* data preparing */
   for (i = 0; i < n; i ++)
   {
      for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
      {
         column = A_j[jj];
         full[i*n+column] = A_data[jj];
      }
      u_data[i] = f_data[i];
   }

   /* Direct solve: use gaussian elimination with pivoting */
   error_flag = fsls_gselim_piv(full, u_data, n);
   
   return (error_flag);
}

/*!
 * \fn JX_Int fsls_AMGRelaxGS
 * \brief Gauss-Seidel type relaxation
 * \param *A pointer to the matrix to be relaxed
 * \param *f pointer to the rhs vector
 * \param *cf_marker pointer to the CF-marker 
 * \param relax_order relaxation order
 * \param *u pointer to the approxiation 
 * \note The iterative scheme of weighted Jacobi for 'Au = f' 
 *  can be described as follows:
 *   if a_{ii} != 0
 *   u_i^{k+1} = (1/a_{ii})*( f_i - \sum_{j=1}^{i-1}a_{ij}*u_j^{k+1} - \sum_{j=i+1}^{n}a_{ij}*u_j^{k} ) 
 * \note Attention, 'cf_marker' has been served as a fine-to-coarse mapping when constructing
 *       interpolation, so, for now, cf_marker[i] >= 0 indicates 'i' is a C-point. 2011/04/14
 * \author peghoty
 * \date 2010/08/30
 */
JX_Int 
fsls_AMGRelaxGS( fsls_CSRMatrix  *A,
                 fsls_Vector     *f,
                 JX_Int             *cf_marker,
                 JX_Int              relax_order,
                 fsls_Vector     *u )
{
   JX_Int      n       = fsls_CSRMatrixNumRows(A);
   JX_Real  *A_data  = fsls_CSRMatrixData(A);
   JX_Int     *A_i     = fsls_CSRMatrixI(A);
   JX_Int     *A_j     = fsls_CSRMatrixJ(A);

   JX_Real  *u_data  = fsls_VectorData(u);
   JX_Real  *f_data  = fsls_VectorData(f);
   	          
   JX_Int      i,ii,jj;
   JX_Int      err_flag = 0;
   JX_Real   res  = 0.0;
   JX_Real   zero = 0.0;

   if (relax_order == ASCEND)  /* Relax all points ascendingly */
   {
             for (i = 0; i < n; i ++)
             {
                /* If diagonal is nonzero, relax point i; otherwise, skip it */
                if (A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }
   else if (relax_order == DESCEND)  /* Relax all points descendingly */
   {
             for (i = n-1; i > -1; i --)
             {
                /* If diagonal is nonzero, relax point i; otherwise, skip it */
                if (A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }   
   else if (relax_order == FPFIRST)  /* F points first, then C points */
   {
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
             
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == CPFIRST)  /* C points first, then F points */
   {
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
             
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == FP_ONLY)  /* only F points be relaxed */
   {
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }
   else if (relax_order == CP_ONLY)  /* only C points be relaxed */
   {
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }         
   else
   {
            jx_printf("\n >>> \033[31mWaring:\033[00m wrong relaxation order!!\n\n");
            err_flag = -1;
            return (err_flag);
   }

   return (err_flag);       
}


/*!
 * \fn JX_Int fsls_AMGRelaxGS_Partial
 * \brief Gauss-Seidel type relaxation
 * \param *A pointer to the matrix to be relaxed
 * \param *f pointer to the rhs vector
 * \param *cf_marker pointer to the CF-marker 
 * \param relax_order relaxation order
 * \param begin the starting index
 * \param end the ending index
 * \param *u pointer to the approxiation 
 * \note The iterative scheme of weighted Jacobi for 'Au = f' 
 *  can be described as follows:
 *   if a_{ii} != 0
 *   u_i^{k+1} = (1/a_{ii})*( f_i - \sum_{j=1}^{i-1}a_{ij}*u_j^{k+1} - \sum_{j=i+1}^{n}a_{ij}*u_j^{k} )  
 * \author peghoty
 * \date 2010/12/18
 */
JX_Int 
fsls_AMGRelaxGS_Partial( fsls_CSRMatrix  *A,
                         fsls_Vector     *f,
                         JX_Int             *cf_marker,
                         JX_Int              relax_order,
                         JX_Int              begin,
                         JX_Int              end,
                         fsls_Vector     *u )
{
   JX_Real  *A_data  = fsls_CSRMatrixData(A);
   JX_Int     *A_i     = fsls_CSRMatrixI(A);
   JX_Int     *A_j     = fsls_CSRMatrixJ(A);

   JX_Real  *u_data  = fsls_VectorData(u);
   JX_Real  *f_data  = fsls_VectorData(f);
   	          
   JX_Int      i,ii,jj;
   JX_Int      err_flag = 0;
   JX_Real   res  = 0.0;
   JX_Real   zero = 0.0;

   if (relax_order == ASCEND)  /* Relax all points ascendingly */
   {
             for (i = begin; i <= end; i ++)
             {
                /* If diagonal is nonzero, relax point i; otherwise, skip it */
                if (A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }
   else if (relax_order == DESCEND)  /* Relax all points descendingly */
   {
             for (i = end; i >= begin; i --)
             {
                /* If diagonal is nonzero, relax point i; otherwise, skip it */
                if (A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }   
   else if (relax_order == FPFIRST)  /* F points first, then C points */
   {
             /* F points relaxing */
             for (i = begin; i <= end; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
             
             /* C points relaxing */
             for (i = begin; i <= end; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == CPFIRST)  /* C points first, then F points */
   {
             /* C points relaxing */
             for (i = begin; i <= end; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
             
             /* F points relaxing */
             for (i = begin; i <= end; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == FP_ONLY)  /* only F points be relaxed */
   {
             /* F points relaxing */
             for (i = begin; i <= end; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }
   else if (relax_order == CP_ONLY)  /* only C points be relaxed */
   {
             /* C points relaxing */
             for (i = begin; i <= end; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = res / A_data[A_i[i]];
                }
             }
   }         
   else
   {
            jx_printf("\n >>> \033[31mWaring:\033[00m wrong relaxation order!!\n\n");
            err_flag = -1;
            return (err_flag);
   }

   return (err_flag);       
} 

/*!
 * \fn JX_Int fsls_AMGRelaxILUp
 * \brief ILUp smoother in the Cycle process of AMG
 * \param level integer, the number of grid level
 * \param *relax_ilu_data fsls_RelaxILUData object for ILU relaxation
 * \param *A the pointer to the matrix to be solved
 * \param *f the pointer to the rhs vector
 * \param *u the pointer to the approxiation
 * \param *r the pointer to the working vector
 * \author peghoty
 * \date 2010/12/07
 */
JX_Int  
fsls_AMGRelaxILUp( JX_Int                 level,
                   fsls_RelaxILUData  *relax_ilu_data,
                   fsls_CSRMatrix     *A,
                   fsls_Vector        *f,
                   fsls_Vector        *u,
                   fsls_Vector        *r )
{
   JX_Int    n = fsls_CSRMatrixNumRows(A);
   JX_Int    i,j,col;
   JX_Real tmp = 0.0;
         
   JX_Int    *index = fsls_RelaxILUDataIndexArray(relax_ilu_data)[level];
   JX_Real *value = fsls_RelaxILUDataValueArray(relax_ilu_data)[level];
   JX_Real *work  = fsls_RelaxILUDataWork(relax_ilu_data);
   
   JX_Real *z_data = work;
   JX_Real *e_data = work + n;
   JX_Real *r_data = fsls_VectorData(r);   
   JX_Real *u_data = fsls_VectorData(u);      
           
   //----------------------------------------
   //  Compute the residual r = f - A*u 
   //----------------------------------------
   
   fsls_SeqVectorCopy(f, r);
   fsls_CSRMatrixMatvec01(-1.0, A, u, 1.0, r);
  
   //-------------------------------------------------------------
   // forward sweep: solve unit lower matrix equation L*z = r 
   //-------------------------------------------------------------
   
   z_data[0] = r_data[0];
   for (i = 1; i < n; i ++)
   {
      tmp = r_data[i];
      for (j = index[i]; j < index[i+1]; j ++)
      {
         col = index[j];
         if (col < i)
            tmp -= value[j]*z_data[col];
         else
            break;
      }
      z_data[i] = tmp;
   }

   //-------------------------------------------------------------
   // backward sweep: solve upper matrix equation U*e = z 
   //-------------------------------------------------------------         

   e_data[n-1] = z_data[n-1]*value[n-1];
   for (i = n-2; i >= 0; i --)
   {
      tmp = z_data[i];
      for (j = index[i+1]-1; j >= index[i]; j --)
      {
         col = index[j];
         if (col > i)
            tmp -= value[j]*e_data[col];
         else
            break;
      }
      e_data[i] = tmp*value[i];
   } 

   //---------------------------------
   // Correction:  u := u + e  
   //---------------------------------

   for (i = 0; i < n; i ++)
   {
      u_data[i] += e_data[i];
   }

   return (0);
}

/*!
 * \fn JX_Int fsls_AMGRelaxJacobi
 * \brief Jacobi type relaxation
 * \param *A pointer to the matrix to be relaxed
 * \param *f pointer to the rhs vector
 * \param *cf_marker pointer to the CF-marker 
 * \param relax_order relaxation order
 * \param relax_weight relaxation weight
 * \param *u pointer to the approxiation
 * \param *Vtemp pointer to the temporary vector  
 * \note The iterative scheme of weighted Jacobi for 'Au = f' 
 * can be described as follows:
 *   if a_{ii} != 0
 *   u_i^{k+1} = (1-w)*u_i^{k} + w*[ (f_i-\sum_{j!=i}a_{ij}*u_j^{k}) / a_{ii}] 
 *   where w is the relax_weight. Especially, it becomes general Jacobi when w = 1.0.
 * \note Theoreticly speaking, Jacobi relaxation is essentially parallel,
 *       i.e., all the new iterative components can be computed independly
 *       by using the previous iterative vector. So, in this sense, as the 
 *       relaxation order is concerned, there should be no difference among
 *       NATURAL, CPFIRST and FPFIRST. But pay attension, a small modification
 *       has been done here, as for CPFIRST and FPFIRST, new C-wise(F-wise)  
 *       iterative components are used when doing the F-wise(C-wise) relaxation.
 *       This two kinds of relaxations seem like something between Jacobi and 
 *       Gauss-Seidel(Mixed Type). 
 * \note Attention, 'cf_marker' has been served as a fine-to-coarse mapping when constructing
 *       interpolation, so, for now, cf_marker[i] >= 0 indicates 'i' is a C-point. 2011/04/14    
 * \author peghoty
 * \date 2010/08/30
 */
JX_Int 
fsls_AMGRelaxJacobi( fsls_CSRMatrix  *A,
                     fsls_Vector     *f,
                     JX_Int             *cf_marker,
                     JX_Int              relax_order,
                     JX_Real           relax_weight,
                     fsls_Vector     *u,
                     fsls_Vector     *Vtemp )
{
   JX_Int      n       = fsls_CSRMatrixNumRows(A);
   JX_Real  *A_data  = fsls_CSRMatrixData(A);
   JX_Int     *A_i     = fsls_CSRMatrixI(A);
   JX_Int     *A_j     = fsls_CSRMatrixJ(A);

   JX_Real  *u_data     = fsls_VectorData(u);
   JX_Real  *f_data     = fsls_VectorData(f);
   JX_Real  *Vtemp_data = fsls_VectorData(Vtemp);
   	          
   JX_Int      i,ii,jj;
   JX_Int      err_flag = 0;
   JX_Real   res  = 0.0;
   JX_Real   zero = 0.0;
   JX_Real   one_minus_weight = 1.0 - relax_weight;

   if (relax_order == ASCEND || relax_order == DESCEND)  /* Relax all points in natural order */
   {
             /* Copy current approximation into temporary vector */
             for (i = 0; i < n; i ++) Vtemp_data[i] = u_data[i];
             
             for (i = 0; i < n; i ++)
             { 
                /* If diagonal is nonzero, relax point i; otherwise, skip it */
                if (A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * Vtemp_data[ii];
                   }
                   u_data[i] *= one_minus_weight;
                   u_data[i] += relax_weight * res / A_data[A_i[i]];
                }
             }
   }
   else if (relax_order == FPFIRST)  /* F points first, then C points */
   {
             /* Copy current approximation into temporary vector */
             for (i = 0; i < n; i ++) Vtemp_data[i] = u_data[i];
                      
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * Vtemp_data[ii];
                   }
                   u_data[i] *= one_minus_weight;
                   u_data[i] += relax_weight * res / A_data[A_i[i]];
                }
             } 
             
             /* Copy current approximation into temporary vector */
             for (i = 0; i < n; i ++) Vtemp_data[i] = u_data[i];
             
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * Vtemp_data[ii];
                   }
                   u_data[i] *= one_minus_weight;
                   u_data[i] += relax_weight * res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == CPFIRST)  /* C points first, then F points */
   {
             /* Copy current approximation into temporary vector */
             for (i = 0; i < n; i ++) Vtemp_data[i] = u_data[i];
                      
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * Vtemp_data[ii];
                   }
                   u_data[i] *= one_minus_weight;
                   u_data[i] += relax_weight * res / A_data[A_i[i]];
                }
             } 
             
             /* Copy current approximation into temporary vector */
             for (i = 0; i < n; i ++) Vtemp_data[i] = u_data[i];
             
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * Vtemp_data[ii];
                   }
                   u_data[i] *= one_minus_weight;
                   u_data[i] += relax_weight * res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == FP_ONLY)  /* only F points be relaxed */
   {
             /* Copy current approximation into temporary vector */
             for (i = 0; i < n; i ++) Vtemp_data[i] = u_data[i];
                      
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * Vtemp_data[ii];
                   }
                   u_data[i] *= one_minus_weight;
                   u_data[i] += relax_weight * res / A_data[A_i[i]];
                }
             } 
   }
   else if (relax_order == CP_ONLY)  /* only C points be relaxed */
   {
             /* Copy current approximation into temporary vector */
             for (i = 0; i < n; i ++) Vtemp_data[i] = u_data[i];
                      
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * Vtemp_data[ii];
                   }
                   u_data[i] *= one_minus_weight;
                   u_data[i] += relax_weight * res / A_data[A_i[i]];
                }
             } 
   }         
   else
   {
            jx_printf("\n >>> \033[31mWaring:\033[00m wrong relaxation order!!\n\n");
            err_flag = -1;
            return (err_flag);
   }
         
   return (err_flag); 
}

/*!
 * \fn void fsls_RowNormCompute
 * \brief row norm computing for Kaczmarz relaxation
 * \param *A pointer to the matrix
 * \param *x information of row norm for A, v[i] = ||w_i||^2, here w_i
 *        means the vector consist of the entries in i-th row of A. 
 * \author peghoty
 * \date 2010/09/04
 */
void 
fsls_RowNormCompute( fsls_CSRMatrix *A, fsls_Vector *x )
{
   JX_Int      n       = fsls_CSRMatrixNumRows(A);
   JX_Real  *A_data  = fsls_CSRMatrixData(A);
   JX_Int     *A_i     = fsls_CSRMatrixI(A);
   JX_Real  *rownrm  = fsls_VectorData(x);
   JX_Real   tmp     = 0.0;
   JX_Int      i, k;
   
   for (i = 0; i < n; i ++)
   {
      tmp = 0.0;
      for (k = A_i[i]; k < A_i[i+1]; k ++)
      {
          tmp += A_data[k]*A_data[k];
      }
      rownrm[i] = tmp;
   }
}


/*!
 * \fn JX_Int fsls_AMGRelaxKaczmarz
 * \brief Kaczmarz type relaxation
 * \param *A pointer to the matrix to be relaxed
 * \param *f pointer to the rhs vector
 * \param *cf_marker pointer to the CF-marker 
 * \param relax_order relaxation order
 * \param relax_weight relaxation weight
 * \param *u pointer to the approxiation 
 * \param *Vtemp information of row norm for A, v[i] = ||w_i||^2, here w_i
 *        means the vector consist of the entries in i-th row of A. 
 * \note Attention, 'cf_marker' has been served as a fine-to-coarse mapping when constructing
 *       interpolation, so, for now, cf_marker[i] >= 0 indicates 'i' is a C-point. 2011/04/14 
 * \author peghoty
 * \date 2010/09/04
 */
JX_Int 
fsls_AMGRelaxKaczmarz( fsls_CSRMatrix  *A,
                       fsls_Vector     *f,
                       JX_Int             *cf_marker,
                       JX_Int              relax_order,
                       JX_Real           relax_weight,
                       fsls_Vector     *u,
                       fsls_Vector     *Vtemp )
{
   JX_Int      n       = fsls_CSRMatrixNumRows(A);
   JX_Real  *A_data  = fsls_CSRMatrixData(A);
   JX_Int     *A_i     = fsls_CSRMatrixI(A);
   JX_Int     *A_j     = fsls_CSRMatrixJ(A);

   JX_Real  *u_data  = fsls_VectorData(u);
   JX_Real  *f_data  = fsls_VectorData(f);
   JX_Real  *rownrm  = fsls_VectorData(Vtemp);
   	          
   JX_Int      i,k;
   JX_Int      err_flag = 0;
   JX_Real   beta = 0.0;
   
   if (relax_order == ASCEND)  /* Relax all points ascendingly */
   {   
            for (i = 0; i < n; i ++)
            {
               beta = 0.0;
               for (k = A_i[i]; k < A_i[i+1]; k ++)
               {
                  beta += A_data[k]*u_data[A_j[k]];
               }
               if (rownrm[i]) 
                  beta = (f_data[i] - beta) / rownrm[i]; 
               else 
                  beta = 0.0; // reasonable or not?
               for (k = A_i[i]; k < A_i[i+1]; k ++)
               {
                  u_data[A_j[k]] += relax_weight*beta*A_data[k];
               }
            } // end for i 
   }			
   else if (relax_order == DESCEND)  /* Relax all points descendingly */
   { 
            for (i = n-1; i >= 0; i --)
            {
               beta = 0.0;
               for (k = A_i[i]; k < A_i[i+1]; k ++)
               {
                  beta += A_data[k]*u_data[A_j[k]];
               }
               if (rownrm[i]) 
                  beta = (f_data[i] - beta) / rownrm[i]; 
               else 
                  beta = 0.0; // reasonable or not?
               for (k = A_i[i]; k < A_i[i+1]; k ++)
               {
                  u_data[A_j[k]] += relax_weight*beta*A_data[k];
               }
            } // end for i 
   }
   else if (relax_order == FPFIRST)  /* F points first, then C points */
   { 
            for (i = 0; i < n; i ++)
            {
               if (cf_marker[i] < 0)
               {
                  beta = 0.0;
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     beta += A_data[k]*u_data[A_j[k]];
                  }
                  if (rownrm[i]) 
                     beta = (f_data[i] - beta) / rownrm[i]; 
                  else 
                     beta = 0.0; // reasonable or not?
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     u_data[A_j[k]] += relax_weight*beta*A_data[k];
                  }
               }
            } // end for i
            for (i = 0; i < n; i ++)
            {
               if (cf_marker[i] >= 0)
               {
                  beta = 0.0;
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     beta += A_data[k]*u_data[A_j[k]];
                  }
                  if (rownrm[i]) 
                     beta = (f_data[i] - beta) / rownrm[i]; 
                  else 
                     beta = 0.0; // reasonable or not?
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     u_data[A_j[k]] += relax_weight*beta*A_data[k];
                  }
               }
            } // end for i              
   }
   else if (relax_order == CPFIRST)  /* C points first, then F points */
   { 
            for (i = 0; i < n; i ++)
            {
               if (cf_marker[i] >= 0)
               {
                  beta = 0.0;
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     beta += A_data[k]*u_data[A_j[k]];
                  }
                  if (rownrm[i]) 
                     beta = (f_data[i] - beta) / rownrm[i]; 
                  else 
                     beta = 0.0; // reasonable or not?
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     u_data[A_j[k]] += relax_weight*beta*A_data[k];
                  }
               }
            } // end for i  
            for (i = 0; i < n; i ++)
            {
               if (cf_marker[i] < 0)
               {
                  beta = 0.0;
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     beta += A_data[k]*u_data[A_j[k]];
                  }
                  if (rownrm[i]) 
                     beta = (f_data[i] - beta) / rownrm[i]; 
                  else 
                     beta = 0.0; // reasonable or not?
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     u_data[A_j[k]] += relax_weight*beta*A_data[k];
                  }
               }
            } // end for i
   } 
   else if (relax_order == FP_ONLY)  /* only F points are relaxed */
   {  
            for (i = 0; i < n; i ++)
            {
               if (cf_marker[i] < 0)
               {
                  beta = 0.0;
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     beta += A_data[k]*u_data[A_j[k]];
                  }
                  if (rownrm[i]) 
                     beta = (f_data[i] - beta) / rownrm[i]; 
                  else 
                     beta = 0.0; // reasonable or not?
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     u_data[A_j[k]] += relax_weight*beta*A_data[k];
                  }
               }
            } // end for i
   } 
   else if (relax_order == CP_ONLY)  /* only C points are relaxed */
   { 
            for (i = 0; i < n; i ++)
            {
               if (cf_marker[i] >= 0)
               {
                  beta = 0.0;
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     beta += A_data[k]*u_data[A_j[k]];
                  }
                  if (rownrm[i]) 
                     beta = (f_data[i] - beta) / rownrm[i]; 
                  else 
                     beta = 0.0; // reasonable or not?
                  for (k = A_i[i]; k < A_i[i+1]; k ++)
                  {
                     u_data[A_j[k]] += relax_weight*beta*A_data[k];
                  }
               }
            } // end for i
   }           
   else
   {
            jx_printf("\n >>> \033[31mWaring:\033[00m wrong relaxation order!!\n\n");
            err_flag = -1;
            return (err_flag);
   }

   return (err_flag);  
}

/*!
 * \fn JX_Int fsls_AMGRelaxPolynomial
 * \brief Polynomial approx to A^{-1} as MG smoother: JK&LTZ2010
 * \param degree degree of polynomial 
 * \param *A pointer to the matrix to be relaxed
 * \param *f pointer to the rhs vector
 * \param *u pointer to the approxiation
 * \param *R pointer to the temporary vector
 * \param *V0 pointer to the temporary vector
 * \param *V pointer to the temporary vector
 * \param *RAV pointer to the temporary vector   
 * \author peghoty
 * \date 2010/09/11
 */
JX_Int 
fsls_AMGRelaxPolynomial( JX_Int              degree,
                         fsls_CSRMatrix  *A,
                         fsls_Vector     *f,
                         fsls_Vector     *u,
                         fsls_Vector     *R,
                         fsls_Vector     *V0,
                         fsls_Vector     *V,
                         fsls_Vector     *RAV )
{
   JX_Int      n       = fsls_CSRMatrixNumRows(A);
   JX_Real  *A_data  = fsls_CSRMatrixData(A);
   JX_Int     *A_i     = fsls_CSRMatrixI(A);
   JX_Int     *A_j     = fsls_CSRMatrixJ(A);

   JX_Real  *R_data   = fsls_VectorData(R);
   JX_Real  *V_data   = fsls_VectorData(V);
   JX_Real  *V0_data  = fsls_VectorData(V0);
   JX_Real  *RAV_data = fsls_VectorData(RAV);
   	          
   JX_Int      i,j;
   JX_Int      err_flag = 0;

   JX_Real   smaxa;
   JX_Real   smina;
   JX_Real   smu0;
   JX_Real   smu1;
   JX_Real   skappa;
   JX_Real   delta;
   JX_Real   delta2;
   JX_Real   sm;
   JX_Real   sm01;
   JX_Real   smsqrt;
   JX_Real   chi;
   JX_Real   snj;
   JX_Real   ARi;

  /*--------------------------------------------------------
   *  parameter computing
   *-------------------------------------------------------*/
   
   smaxa  = fsls_CSRMatrixInfiniteNorm(A);
   smina  = smaxa / 4.0;
   smu0   = 1.0 / smaxa;
   smu1   = 1.0 / smina;
   skappa = sqrt(smaxa / smina);
   delta  = (skappa - 1.0) / (skappa + 1.0);
   delta2 = delta*delta;
   sm     = 0.5*(smu0 + smu1);
   sm01   = smu0*smu1;
   smsqrt = sm + sqrt(sm01);
   chi    = 2*sm01 / smsqrt;
   
   
  /*--------------------------------------------------------
   *  polynomial relaxation process
   *-------------------------------------------------------*/
   
   fsls_SeqVectorCopy(f, R);
   fsls_CSRMatrixMatvec(-1.0, A, u, 1.0, R);  // R = f - A*u
   
   for (i = 0; i < n; i ++)
   {
      ARi = 0.0;
      for (j = A_i[i]; j < A_i[i+1]; j ++)
      {
         ARi += A_data[j]*R_data[A_j[j]];
      }
      V0_data[i] = sm*R_data[i];
      V_data[i]  = smsqrt*R_data[i] - sm01*ARi;
   }
   
   for (i = 1; i < degree; i ++)
   {
      fsls_SeqVectorCopy(R, RAV);
      fsls_CSRMatrixMatvec(-1.0, A, V, 1.0, RAV);  // RAV = r - A*V
      
      for (j = 0; j < n; j ++)
      {
         snj = chi*RAV_data[j] + delta2*(V_data[j] - V0_data[j]);
         V0_data[j] = V_data[j];
         V_data[j] += snj;
      }
   }
   
   fsls_SeqVectorAxpy(1.0, V, u); // u = u + V

   return (err_flag); 
}

/*!
 * \fn JX_Int fsls_AMGRelaxSOR
 * \brief Gauss-Seidel type relaxation
 * \param *A pointer to the matrix to be relaxed
 * \param *f pointer to the rhs vector
 * \param *cf_marker pointer to the CF-marker 
 * \param relax_order relaxation order
 * \param relax_weight relaxation weight
 * \param *u pointer to the approxiation
 * \note The iterative scheme of weighted Jacobi for 'Au = f' 
 *  can be described as follows:
 *   if a_{ii} != 0
 *   u_i^{k+1} = (1-w)*u_i^{k+1} + 
                 w*(1/a_{ii})*( f_i - \sum_{j<i}a_{ij}*u_j^{k+1} - \sum_{j>i}a_{ij}*u_j^{k} ) 
 *   where w is the relax_weight. Especially, it becomes general GS when w = 1.0.    
 * \note Attention, 'cf_marker' has been served as a fine-to-coarse mapping when constructing
 *       interpolation, so, for now, cf_marker[i] >= 0 indicates 'i' is a C-point. 2011/04/14               
 * \author peghoty
 * \date 2010/08/30
 */
JX_Int 
fsls_AMGRelaxSOR( fsls_CSRMatrix  *A,
                  fsls_Vector     *f,
                  JX_Int             *cf_marker,
                  JX_Int              relax_order,
                  JX_Real           relax_weight,
                  fsls_Vector     *u  )
{
   JX_Int      n       = fsls_CSRMatrixNumRows(A);
   JX_Real  *A_data  = fsls_CSRMatrixData(A);
   JX_Int     *A_i     = fsls_CSRMatrixI(A);
   JX_Int     *A_j     = fsls_CSRMatrixJ(A);

   JX_Real  *u_data  = fsls_VectorData(u);
   JX_Real  *f_data  = fsls_VectorData(f);
   	          
   JX_Int      i,ii,jj;
   JX_Int      err_flag = 0;
   JX_Real   res  = 0.0;
   JX_Real   zero = 0.0;
   JX_Real   one_minus_weight = 1.0 - relax_weight;   

   if (relax_order == ASCEND)  /* Relax all points ascendingly */
   {
             for (i = 0; i < n; i ++)
             {
                /* If diagonal is nonzero, relax point i; otherwise, skip it */
                if (A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }
   }
   else if (relax_order == DESCEND)  /* Relax all points descendingly */
   {
             for (i = n-1; i > -1; i --)
             {
                /* If diagonal is nonzero, relax point i; otherwise, skip it */
                if (A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }
   }   
   else if (relax_order == FPFIRST)  /* F points first, then C points */
   {
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }
             
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == CPFIRST)  /* C points first, then F points */
   {
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }
             
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }    
   }
   else if (relax_order == FP_ONLY)  /* only F points be relaxed */
   {
             /* F points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] < 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }
   }
   else if (relax_order == CP_ONLY)  /* only C points be relaxed */
   {
             /* C points relaxing */
             for (i = 0; i < n; i ++)
             {
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
                if (cf_marker[i] >= 0 && A_data[A_i[i]] != zero)
                {
                   res = f_data[i];
                   for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
                   {
                      ii = A_j[jj];
                      res -= A_data[jj] * u_data[ii];
                   }
                   u_data[i] = one_minus_weight*u_data[i] + relax_weight*res / A_data[A_i[i]];
                }
             }
   }         
   else
   {
            jx_printf("\n >>> \033[31mWaring:\033[00m wrong relaxation order!!\n\n");
            err_flag = -1;
            return (err_flag);
   }

   return (err_flag);       
}   
