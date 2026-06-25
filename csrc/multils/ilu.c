/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 * ilu.c
 *
 * Created by peghoty 2010/12/07
 * Xiangtan University
 * peghoty@163.com
 *  
 */

#include "jx_multils.h"

JX_Int
fsls_ILUp_FillInPredict( fsls_CSRMatrix *A, JX_Int p )
{
   JX_Int fillins = 0; 

   switch (p)
   {
      case 0:
      {
          fillins = 0;
      }
      break;
       
      case 1:
      {
         fillins = fsls_ilu1_FillInPredict(A);
      }
      break;
   
      case 2:
      {
         fillins = fsls_ilu2_FillInPredict(A);
      }
      break;
      
      default:
      {
         fillins = fsls_ilup_FillInPredict(A, p);
      }
      break; 
   }
   
   return fillins;
}

/*!
 * \fn JX_Int fsls_ilu1_FillInPredict
 * \brief Predict the total number of fill-in's during the ILU(1) decomposition for A. 
 *        The corresponding algorithom is based on Graph Theory.
 * \param *A pointer to the matrix to be decomposed
 * \return fillins the possible number of fill-in's
 * \note The diagonals should be first stored in each row.
 * \author Xu Senlin and peghoty
 * \date 2010/03/28
 */
JX_Int
fsls_ilu1_FillInPredict( fsls_CSRMatrix *A )                                
{
   /* information of the matrix */
   JX_Int  n  = fsls_CSRMatrixNumRows(A);
   JX_Int *ia = fsls_CSRMatrixI(A);
   JX_Int *ja = fsls_CSRMatrixJ(A); 
   
  /*----------------------------------------------------------------- 
   *  For a fix row Seed, an auxiliary array 'flag' is needed:
   *
   *  flag[j],j=0(1)n-1: 
   *
   *          / = Seed, if (Seed,j) is a nonzero or a fill-in.   
   *  flag[j] 
   *          \ < Seed, if (Seed,j) is a zero entries.
   *
   *----------------------------------------------------------------*/   
   JX_Int *flag = fsls_CTAlloc(JX_Int,n);   
   
   /* local variables */
   JX_Int j,k1,k2; 
   JX_Int Seed,V1,V2;   // Seed -> V1 -> V2 will be accounted if (V1 < Seed, and V1 < V2) 
   JX_Int fillins = 0;  // counting the total nonzeros after ILU(1) decomposition 
    
   //=========================================================//
   //  Main loop: counting 'fillins' row by row.              //
   //                                                         //
   //  Remark: we skip the first row since no any             //
   //          fill-in's would be generated.                  //
   //=========================================================//

   for (Seed = 1; Seed < n; Seed ++)
   {
      //----------------------------------------------
      //   Initialize the auxiliary array 'flag'. 
      //----------------------------------------------
      
      for (j = ia[Seed]; j < ia[Seed+1]; j ++ )
      { 
         flag[ja[j]] = Seed; 
      }

      //------------------------------------------------------------------
      //  Find all fill-in's.
      //------------------------------------------------------------------
      
      /* Loop for all the nodes in the Dependence Set of node 'Seed'. */
      for (k1 = ia[Seed]+1; k1 < ia[Seed+1]; k1 ++)
      {
         V1 = ja[k1];
         
         if (V1 < Seed)
         {
             /* Loop for all the nodes in the Dependence Set of node 'V1'. */           
             for (k2 = ia[V1]+1; k2 < ia[V1+1]; k2 ++)
             {
                V2 = ja[k2];
                
                if (flag[V2] < Seed && V2 > V1)
                {  
                   flag[V2] = Seed; 
                   fillins ++;   
                }           
             } // end the k2 loop
         } 
      } // end the k1 loop
            
   } // end the Seed loop

   //----------------------------------------------
   //   Free some staff.
   //----------------------------------------------

   fsls_TFree(flag);

   return (fillins);
}

/*!
 * \fn JX_Int fsls_ilu2_FillInPredict
 * \brief Predict the total number of fill-in's during the ILU(2) decomposition for A.
 *        The corresponding algorithom is based on Graph Theory.
 * \param *A pointer to the matrix to be decomposed
 * \return fillins the possible number of fill-in's
 * \author Xu Senlin, peghoty
 * \date 2010/03/28
 */
JX_Int
fsls_ilu2_FillInPredict( fsls_CSRMatrix *A )                                
{
   /* information of the matrix */
   JX_Int  n  = fsls_CSRMatrixNumRows(A);
   JX_Int  nz = fsls_CSRMatrixNumNonzeros(A);   
   JX_Int *ia = fsls_CSRMatrixI(A);
   JX_Int *ja = fsls_CSRMatrixJ(A); 

  /*------------------------------------------------------------------------------ 
   *  For a fix row i, the following local auxiliary arrays are allocated:
   *
   *  (1) index[j],j=0(1)n-1: 
   *
   *             / k, if a_{i,k} is a nonzero(or fill-in) in row i, j means 
   *                  a_{i,k} is the j-th nonzero(or fill-in) in row i.  
   *  index[j] =
   *             \ 0, else
   *
   *  remark: in the array 'index', the column numbers of (original)nonzeros 
   *          are stored first and then those of the fill-in's.
   *
   *  (2) place[j],j=0(1)n-1: 
   *
   *             / k, if a_{i,j} is a nonzero(or fill-in) in row i, k is the 
   *                  position where j is stored in 'index'(i.e.,index[k]=j).   
   *  place[j] =
   *             \ n, else. here, n denotes the number of rows of A.
    
   *---------------------------------------------------------------------------*/   

   JX_Int *work  = fsls_CTAlloc(JX_Int, 2*n);  
   JX_Int *index = work;
   JX_Int *place = work + n; 

   /* local variables */
   JX_Int j;
   JX_Int fillins;   // number of total fill-in's 
   JX_Int rownzfi;   // number of original nonzeros and new generated fill-in's in a fixed row
   JX_Int rownz;     // number of original nonzeros in a fixed row
   JX_Int rownz0;    // number of original nonzeros in the first row(row 0)
   JX_Int Seed;      // starting node for the fill path searching
   JX_Int V1,V2,V3;  
   JX_Int k1,k2,k3;
  
   //---------------------------------
   // Initialization 
   //---------------------------------
    
   fillins = 0;
   rownz0  = ia[1] - ia[0];
   for (j = 0; j < n; j ++)
   { 
      place[j] = n;  
   }
   

   //------------------------------------------------------------
   //  Main loop: counting 'fillins'(contain original nonzeros)
   //             row by row.
   //
   //  Remark: we skip the first row since no any 
   //          fill-in's would be generated.
   //------------------------------------------------------------
   
   for (Seed = 1; Seed < n; Seed ++)
   {
     /*------------------------------------------------------
      *  Initialize the auxiliary arrays. 
      *-----------------------------------------------------*/
      rownz = 0; 
      for (j = ia[Seed]; j < ia[Seed+1]; j ++)
      {
         index[rownz] = ja[j]; 
         place[ja[j]] = rownz;
         rownz ++;
      }
      rownzfi = rownz;
      
     /*------------------------------------------------------
      *  Deal with the nodes on the first circle  
      *  centered around the node 'Seed'.
      *-----------------------------------------------------*/ 
      for (k1 = ia[Seed]+1; k1 < ia[Seed+1]; k1 ++)
      {
           V1 = ja[k1];
           
           if (V1 < Seed)
           {
              /*------------------------------------------------------
               *  Deal with the nodes on the second circle  
               *  centered around the node 'Seed'.
               *-----------------------------------------------------*/           
               for (k2 = ia[V1]+1; k2 < ia[V1+1]; k2 ++)
               {
                  V2 = ja[k2]; 
                  
                  /* node 'V2' hasn't appeared on the first circle */
                  if (place[V2] >= rownz) 
                  { 
                     
                     if (V2 > V1 && place[V2] == n)  
                     {  
                        /* 'place[V2] = n' means (Seed,V2) would generate 
                            a new fill-in (for the first time). */
                        index[rownzfi] = V2;
                        place[V2]      = rownzfi; 
                        rownzfi ++;
                     }
                     
                     if (V2 < Seed)
                     {   
                       
                       /*------------------------------------------------------
                        *  Deal with the nodes on the third circle  
                        *  centered around the node 'Seed'.
                        *-----------------------------------------------------*/ 
                        for (k3 = ia[V2]+1; k3 < ia[V2+1]; k3 ++)
                        {
                           V3 = ja[k3]; 
                           if (place[V3] == n && V3 > fsls_max(V1,V2))
                           {
                             /* 'place[V3] = n' means (Seed,V3) would generate 
                                 a new fill-in (for the first time), and 
                                 'V3 > fsls_max(V1,V2)' to make sure 'Seed->V1->V2->V3' 
                                 constructs a fill path */
                              index[rownzfi] = V3;
                              place[V3]      = rownzfi;
                              rownzfi ++;   
                           }
                        }  // end of k3 loop 
                     }  
                  } 
               } // end of k2 loop 
           } 
      } // end of k1 loop
       
      fillins += rownzfi; 
 
     /*--------------------------------------------------------------------
      *  Recover the 'place' array to make sure: place[i] = n, i=0(1)n-1
      *-------------------------------------------------------------------*/ 
      for (j = 0; j < rownzfi; j ++)
      { 
         place[index[j]] = n; 
      } 
    
   } // end of main loop

   //--------------------------------------------
   //  Free the auxiliary arrays.
   //--------------------------------------------    
    
   fsls_TFree(work);
   
   //--------------------------------------------
   //  Return the true number of fill-in's.
   //-------------------------------------------- 

   return (fillins + rownz0 - nz);
}

/*!
 * \fn JX_Int fsls_ilup_FillInPredict
 * \brief Predict the total number of fill-in's during the ILU(p) decomposition for A.
 *        The corresponding algorithom is implemented purely algebraicly.
 * \param *A pointer to the matrix to be ILU(p)-decomposed
 * \param p the "level of fill-in" parameter (p should be larger than one) 
 * \return fillins the number of possible fill-in's during the ILU(p) decomposition for A.
 *   ierr =  0  --> successful return.
 *   ierr = -1  --> Error, the input matrix may be wrong because
 *                  the elimination process has generated a row
 *                  in L or U whose length is greater than n. 
 *   ierr = -4  --> Illegal value for 'p'.
 * \note (1) All the diagonal elements of the input matrix A must be nonzero.
 *       (2) For better efficiency, you'd better remove all
 *           zeros in the matrix A by using the function 
 *           'fsls_CSRMatrixDeleteZeros(A,0.0)' previous,
 *           otherwise, the statement 'if (val == 0.0) continue;'
 *           in the function is quite necessary, this is why 'a'
 *           is needed in the code.
 *       (3) p should be larger than one.
 * \author peghoty
 * \date 2010/03/10
 */
JX_Int
fsls_ilup_FillInPredict( fsls_CSRMatrix *A, JX_Int p )
{
   /* information of the matrix */
   JX_Int     n  = fsls_CSRMatrixNumRows(A);
   JX_Int    *ia = fsls_CSRMatrixI(A);
   JX_Int    *ja = fsls_CSRMatrixJ(A);
   JX_Real *a  = fsls_CSRMatrixData(A);
   
  /*-----------------------------------------------------------------------------
   *  Auxiliary arrays
   *
   *  1. For the current row, the following local auxiliary arrays are allocated:
   *
   *  (1) index[i],i=0(1)n-1: be used to store the column numbers of all 
   *                          nonzeros in the current row.
   *  (2) level[i],i=0(1)n-1: be used to store the level of fill-in values
   *                          of all nonzeros in the current row.
   *  (3) place[i],i=0(1)n-1: the position of a_{*,i} in 'value' or 'index'.
   *                          
   *  Attention: the above arrays are used in the compressing way, they 
   *             are divided into the Left-part(Lower-part) and Right-part
   *             (Upper-part) by the position 'k', i.e., the row number of 
   *             the current row.
   *
   *  (4) count[i],i=0(1)n-1: be used to store the number of fill-in's in the 
   *                          i-th row during the ILU(p) decomposition.
   *
   *  (5) ColandLevel[i][j],i=0(1)n-1,j=0(1)2*Upart: 
   *      (a) ColandLevel[i][0] = Upart: number of U-part entries(excluding
   *          the diagonal) in row i.
   *      (b) ColandLevel[i][j],j=1(1)Upart: column number of the j-th U-part  
   *          entry(excluding the diagonal) in row i.
   *      (c) ColandLevel[i][j],j=Upart+1(1)2*Upart: level-value of the j-th   
   *          U-part entry(excluding the diagonal) in row i.                       
   *-----------------------------------------------------------------------------*/

   JX_Int  *index = NULL;    
   JX_Int  *place = NULL;    
   JX_Int  *level = NULL; 
   JX_Int  *count = NULL;
   JX_Int  *work  = NULL;      
   JX_Int **ColandLevel = NULL;   
   
   /* local variables */
   JX_Int j,ierr = 0;
   JX_Int row,col;
   JX_Int pos,upos;
   JX_Int numL;     // for counting the nonzeros in the L-part of the current row
   JX_Int numU;     // for counting the nonzeros in the U-part of the current row
   JX_Int GE;       // for counting the rows that contribute in the GE of the current row
   JX_Int flStartU; // record the starting position for U-part fill-ins in the current row
   JX_Int flEndL;   // record the ending position for L-part fill-ins in the current row
   JX_Int flEndU;   // record the ending position for U-part fill-ins in the current row
   JX_Int fillins;  // number of possible fill-in's during the LU decomposition for A.
   JX_Int Upart;    // number of U-part entry(excluding the diagonal) in the current row
   JX_Int levLTpU;  // the number of U-part fill-in's whose level-value is 
                 // less than 'p' in the current row
   JX_Int mincol,minpos;
   JX_Int tmpindex,tmplevel;
   JX_Int GElevel;
   JX_Int cnt,lev;
   JX_Real val;

  /*-------------------------------------------------------
   *  Is the parameter 'p' legal?
   *------------------------------------------------------*/
       
   if (p < 2)
   {
      jx_printf("\n >> This function is designed for p >= 2!\n\n");
      ierr = -4;
      return ierr;
   }


  /*-------------------------------------------------------
   *  Allocate memory for the auxiliary arrays
   *------------------------------------------------------*/

   work = fsls_CTAlloc(JX_Int,4*n);
   ColandLevel = fsls_CTAlloc(JX_Int *,n);
   index = work; place = index + n; level = place + n; count = level + n;   
      

  /*-------------------------------------------------------
   *  Initialize 'place'
   *------------------------------------------------------*/   
     
   for (j = 0; j < n; j ++) place[j] = -1;


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++
   *  The main loop of ILU(p) decomposition
   *++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   
   for (row = 0; row < n; row ++)
   {
   
     /*-------------------------------------------------------
      *  Step 1:  Assign the nonzeros of the current row into
      *  L-part and U-part,respectively, and generate all the
      *  working arrays: index,level,place.
      *------------------------------------------------------*/ 
                
      numL = 0;  
      numU = 1;  // start from the diagonal 

      for (j = ia[row]; j < ia[row+1]; j ++)
      {
         val = a[j];   // value
         col = ja[j];  // column number
         
         if (val == 0.0) continue; // skip the zeros in the matrix
      
         if (col < row)
         {
            index[numL] = col;
            level[numL] = 0;
            place[col]  = numL;
            numL ++; 
         }
         else if (col == row)
         {
            index[row] = col;
            level[row] = 0;
            place[col] = row;
         }
         else
         {
            pos = row + numU;
            index[pos] = col;
            level[pos] = 0;
            place[col] = pos;
            numU ++;
         } 
      }  
  
   
     /*-------------------------------------------------------
      *  Step 2:  Eliminate previous rows.
      *------------------------------------------------------*/ 
      
      GE = 0;
      flStartU = row + numU;       
      
      while (GE < numL)
      {

         /*-------------------------------------------------------
          *  Step 2.1  In order to do the elimination in the   
          *  correct order we must select the smallest column
          *  index among index[k], k=GE,...,numL-1.
          *------------------------------------------------------*/
          
          mincol = index[GE];
          minpos = GE;
          
          for (j = GE+1; j < numL; j ++)
          {
             if (index[j] < mincol)
             {
                mincol = index[j];
                minpos = j;
             }
          }
          
          
         /*-------------------------------------------------------
          *  Step 2.2  if (minpos != GE), do the exchanging in
          *  the following arrays: index,place,level.
          *------------------------------------------------------*/
          
          if (minpos != GE)
          {
             /* swap in "index" */
             tmpindex = index[GE];
             index[GE] = index[minpos];
             index[minpos] = tmpindex;
             
             /* swap in "place" */
             place[mincol]   = GE;
             place[tmpindex] = minpos;
             
             /* swap in "level" */
             tmplevel = level[GE];
             level[GE] = level[minpos];
             level[minpos] = tmplevel;            
          } 


         /*-------------------------------------------------------
          *  Step 2.3  Zero out place[mincol].  (L-part) 
          *------------------------------------------------------*/
          place[mincol] = -1;
          
          
         /*-----------------------------------------------------------------------
          *  Step 2.4  Eliminate the current row using the mincol-th row.
          *----------------------------------------------------------------------*/   

          GElevel = level[GE]; // level of fill-in of the GE-th entry in 'value'
          
          if (GElevel > p)
          {
             GE ++;
             continue;
          }
          
          /* update the current row */
          
          // number of U-part entries(excluding the diagonal) in the mincol-th row
          Upart = ColandLevel[mincol][0]; 
          for (j = 1; j <= Upart; j ++)
          {
             col = ColandLevel[mincol][j];       // column number of the j-th U-part entry
             lev = ColandLevel[mincol][Upart+j]; // level of fill-in of the j-th U-part entry
             pos = place[col];
             
             if (col < row)      // dealing with lower part
             {
                if (pos < 0)     // this is a fill-in element
                {
                   if (numL >= n)
                   {
                      jx_printf("\n >> Incomprehensible error, matrix must be wrong!\n\n");
                      ierr = -1;
                      return ierr;
                   }
                   index[numL] = col;
                   level[numL] = GElevel + lev + 1;
                   place[col]  = numL;
                   numL ++;
                }
                else  // this is not a fill-in element
                {
                   level[pos] = fsls_min(level[pos], GElevel + lev + 1);
                }
             }
             else  // dealing with upper part
             {
                if (pos < 0)  // this is a fill-in element
                {
                   if (numU >= n)
                   {
                      jx_printf("\n >> Incomprehensible error, matrix must be wrong!\n\n");
                      ierr = -1;
                      return ierr;
                   }
                   upos = row + numU;
                   index[upos] = col;
                   level[upos] = GElevel + lev + 1;
                   place[col]  = upos;
                   numU ++;
                }
                else  // this is not a fill-in element
                {
                   level[pos] = fsls_min(level[pos], GElevel + lev + 1); 
                }
             }
          }
              
          GE ++;
          
      } // end while loop
   
   
     /*-------------------------------------------------------
      *  Step 3:  Zero out place[].  (U-part) 
      *------------------------------------------------------*/          
          
      for (j = 0; j < numU; j ++)
      {
         place[index[row+j]] = -1;
      }


     /*---------------------------------------------------------------
      *  Step 4:  Find all the fill-in's and get ColandLevel[row]
      *--------------------------------------------------------------*/ 

      flEndL = numL;
      flEndU = row + numU;

      count[row] = 0;

      for (j = flStartU; j < flEndU; j ++)
      {
         if (level[j] <= p) count[row] ++;
      }  
      levLTpU = count[row];
      
      for (j = 0; j < flEndL; j ++)
      {
         if (level[j] > 0 && level[j] <= p) count[row] ++;
      }
     /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      * Remark: here, we can't use such code as follows:
      *
      *   for (j = flStartL; j < flEndL; j ++)
      *   {
      *      if (level[j] <= p) count[row] ++;
      *   }
      *  
      *   where the statement 'flStartL = numL;' should be added
      *   after 'flStartU = row + numU;'.
      *
      *   Why? Don't forget we did many swaps in order to seek
      *   smallest column index, some entry(whose level-value is 
      *   in the interval [1,p]) may have been moved into
      *   level[k],k=0(1)flStartL-1.
      *                                        peghoty 2010/03/10
      *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/
 
     /*---------------------------------------------------------------------
      * Upart: number of U-part entries(excluding the diagonal) 
      *        in the current row, where
      *    (1) flStartU - row - 1: number of original nonzeros in 
      *        the current row of A(excluding the diagonal);
      *    (2) levLTpU: the number of U-part fill-in's whose level-value 
                is less than 'p' in the current row.
      *-------------------------------------------------------------------*/
      Upart = (flStartU - row - 1) + levLTpU;

      ColandLevel[row] = fsls_CTAlloc(JX_Int,2*Upart+1);

      cnt = 0;
      ColandLevel[row][cnt++] = Upart;
      for (j = row+1; j < flEndU; j ++)
      {
         if (level[j] <= p)
         {
            ColandLevel[row][cnt] = index[j];
            ColandLevel[row][Upart+cnt] = level[j];
            cnt ++; 
         }
      }
 
   }  // end the main loop


  /*-----------------------------------------------------------
   *  Count the number of fill-ins during the decomposition.
   *---------------------------------------------------------*/ 

   fillins = 0;
   for (j = 0; j < n; j ++)
   {
      fillins += count[j];
   }
   
   fsls_TFree(work);
   for (j = 0; j < n; j ++)
   {
      fsls_TFree(ColandLevel[j]);
   }
   fsls_TFree(ColandLevel);

   return fillins;
}

/*!
 * \fn JX_Int fsls_ILUp_Decomp
 * \brief ILU(p) decomposition of A.
 * \param *A pointer to the matrix to be ILU(p)-decomposed
 * \param *p the "level of fill-in" parameter
 * \param **indexLU_ptr JX_Real pointer to the index part of the resulting LU decomposition in MSR format
 * \param **valueLU_ptr JX_Real pointer to the value part of the resulting LU decomposition in MSR format
 * \return ierr
 *   ierr =  0  --> successful return.
 *   ierr = -1  --> Error,the input matrix may be wrong because the elimination process
 *                  has generated a row in L or U whose length is greater than n. 
 *   ierr = -2  --> The matrix L overflows the array 'valueLU'.
 *   ierr = -3  --> The matrix U overflows the array 'valueLU'.
 *   ierr = -4  --> Illegal value for 'p'.
 *   ierr = -5  --> Zero diagonal in the input matrix.
 * \note (1) All the diagonal elements of the input matrix A must be nonzero.
 *       (2) For better efficiency, you'd better remove all zeros in the matrix A by using  
 *           the function 'fsls_CSRMatrixDeleteZeros(A,0.0)' previously, otherwise, the statement
 *           'if (val == 0.0) continue;' in the function is quite necessary.
 *       (3) In order to save memory, 'sizeMSR' is predicted at the very beginning by predicting 
 *           the number of possible fill-in's during the decomposition.
 *       (4) For the efficiency in solving the triangular linear system, the first n components in 
 *           indexLU store the reciprocals of the corresponding diagonal entries.
 * \author peghoty
 * \date 2010/03/12, modified 2010/12/07
 */
JX_Int
fsls_ILUp_Decomp( fsls_CSRMatrix  *A, 
                  JX_Int              p,
                  JX_Int            **indexLU_ptr,
                  JX_Real         **valueLU_ptr )
{
   /* information of the matrix */
   JX_Int     n  = fsls_CSRMatrixNumRows(A);
   JX_Int     nz = fsls_CSRMatrixNumNonzeros(A);
   JX_Int    *ia = fsls_CSRMatrixI(A);
   JX_Int    *ja = fsls_CSRMatrixJ(A);
   JX_Real *a  = fsls_CSRMatrixData(A);
   
   /* storage for L and U in MSR format */
   JX_Int     fillins;
   JX_Int     sizeMSR;
   JX_Real *valueLU = NULL;
   JX_Int    *indexLU = NULL;
   
  /*-----------------------------------------------------------------------------
   *  Auxiliary arrays
   *
   *  1. For the current row, the following local auxiliary arrays are allocated:
   *
   *  (1) value[i],i=0(1)n-1: be used to store the values of all nonzeros 
   *                          in the current row.
   *  (2) index[i],i=0(1)n-1: be used to store the column numbers of all 
   *                          nonzeros in the current row.
   *  (3) level[i],i=0(1)n-1: be used to store the level of fill-in values
   *                          of all nonzeros in the current row.
   *  (4) place[i],i=0(1)n-1: the position of a_{*,i} in 'value' or 'index'.
   *                          
   *  Attention: the above arrays are used in the compressing way, they 
   *             are divided into the Left-part(Lower-part) and Right-part
   *             (Upper-part) by the position 'k', i.e., the row number of 
   *             the current row.
   *
   *  2. Another global auxiliary arrays:
   * 
   *  (1) Uhead[i],i=0(1)n-1: be used to record the address where the U-part
   *                          starts in valueLU (indexLU or levelLU) in row i.                       
   *  (2) levelLU[i],i=0(1)sizeMSR-1: be used to store the level of fill-in 
   *                                  values for all the U-part elements in
   *                                  'valueLU'. (there apparently exists 
   *                                  some waste!!)
   *-----------------------------------------------------------------------------*/

   JX_Real *value   = NULL; 
   JX_Int    *index   = NULL;    
   JX_Int    *place   = NULL;    
   JX_Int    *level   = NULL;    
   JX_Int    *Uhead   = NULL;
   JX_Int    *levelLU = NULL;
   JX_Int    *work    = NULL;
   
   /* local variables */
   JX_Int  j,ierr = 0;
   JX_Int  numL,numU;    // for counting the nonzeros in the L- and U-part of the current row, respectively
   JX_Int  GE;           // for counting the rows that contribute in the GE of the current row
   JX_Int  mincol,minpos;
   JX_Int  tmpindex,tmplevel;
   JX_Int  row,col,pos,upos,cntmsr;   
   JX_Int  GElevel;
   JX_Real val;
   JX_Real tmpvalue;
   JX_Real factor;   // elimination factor
   

   //---------------------------------------------
   //  Is the parameter 'p' legal?
   //---------------------------------------------
   
   if (p < 0)
   {
      jx_printf("\n >> Illegal value for p!\n\n");
      ierr = -4; return ierr;
   }


   //---------------------------------------------
   //  Compute the sizeMSR
   //---------------------------------------------
    
   fillins = fsls_ILUp_FillInPredict(A, p);
   sizeMSR = nz + fillins + 1;
   

   //---------------------------------------------
   //  Allocate memories
   //---------------------------------------------
   
   work  = fsls_CTAlloc(JX_Int, sizeMSR + 4*n);
   index = work;      
   place = index + n; level   = place + n;
   Uhead = level + n; levelLU = Uhead + n;
   
   value = fsls_CTAlloc(JX_Real, n);
   valueLU = fsls_CTAlloc(JX_Real, sizeMSR);
   indexLU = fsls_CTAlloc(JX_Int, sizeMSR);
   
  
   //---------------------------------------------
   //  Initialize 'place', cntmsr and indexLU[0]
   //---------------------------------------------
        
   for (j = 0; j < n; j ++) place[j] = -1;
   cntmsr = n + 1;
   indexLU[0] = cntmsr;


   //---------------------------------------------
   //  The main loop of ILU(p) decomposition
   //---------------------------------------------
      
   for (row = 0; row < n; row ++)
   {
   
     /*-------------------------------------------------------
      *  Step 1:  Assign the nonzeros of the current row into
      *  L-part and U-part,respectively, and generate all the
      *  working arrays: value,index,level,place.
      *------------------------------------------------------*/ 
                
      numL = 0;  
      numU = 1;  

      for (j = ia[row]; j < ia[row+1]; j ++)
      {
         val = a[j];   // value
         col = ja[j];  // column number
         
         if (val == 0.0) continue;
      
         if (col < row)
         {
            value[numL] = val;
            index[numL] = col;
            level[numL] = 0;
            place[col]  = numL;
            numL ++; 
         }
         else if (col == row)
         {
            value[row] = val;
            index[row] = col;
            level[row] = 0;
            place[col] = row;
         }
         else
         {
            pos = row + numU;
            value[pos] = val;
            index[pos] = col;
            level[pos] = 0;
            place[col] = pos;
            numU ++;
         } 
      }  
  
   
     /*-------------------------------------------------------
      *  Step 2:  Eliminate previous rows.
      *------------------------------------------------------*/ 
      
      GE = 0;
      
      while (GE < numL)
      {

         /*-------------------------------------------------------
          *  Step 2.1  In order to do the elimination in the   
          *  correct order we must select the smallest column
          *  index among index[k], k=GE,...,numL-1.
          *------------------------------------------------------*/
          
          mincol = index[GE];
          minpos = GE;
          
          for (j = GE+1; j < numL; j ++)
          {
             if (index[j] < mincol)
             {
                mincol = index[j];
                minpos = j;
             }
          }
          
          
         /*-------------------------------------------------------
          *  Step 2.2  if (minpos != GE), do the exchanging in
          *  the following arrays: value,index,place,level.
          *------------------------------------------------------*/
          
          if (minpos != GE)
          {
             /* swap in "index" */
             tmpindex = index[GE];
             index[GE] = index[minpos];
             index[minpos] = tmpindex;
             
             /* swap in "place" */
             place[mincol]   = GE;
             place[tmpindex] = minpos;
             
             /* swap in "level" */
             tmplevel = level[GE];
             level[GE] = level[minpos];
             level[minpos] = tmplevel;

             /* swap in "value" */
             tmpvalue = value[GE];
             value[GE] = value[minpos];
             value[minpos] = tmpvalue;             
          } 


         /*-------------------------------------------------------
          *  Step 2.3  Zero out place[mincol].  (L-part) 
          *------------------------------------------------------*/
          place[mincol] = -1;
          
          
         /*----------------------------------------------------------------
          *  Step 2.4  Eliminate the current row using the mincol-th row.
          *---------------------------------------------------------------*/   
          
          /* get the multiplier and the corresponding level of fill-in */
          factor  = value[GE] / valueLU[mincol];
          GElevel = level[GE];
          
          if (GElevel > p)
          {
             GE ++;
             continue;
          }
          
          /* update the current row */
          value[GE] = factor;
          for (j = Uhead[mincol]; j < indexLU[mincol+1]; j ++)
          {
             val = factor*valueLU[j];
             col = indexLU[j];
             pos = place[col];
             
             if (col < row)      // dealing with lower part
             {
                if (pos < 0)     // this is a fill-in element
                {
                   if (numL >= n)
                   {
                      jx_printf("\n >> Incomprehensible error, matrix must be wrong!\n\n");
                      ierr = -1;
                      return ierr;
                   }
                   value[numL] = -val;
                   index[numL] = col;
                   level[numL] = GElevel + levelLU[j] + 1;
                   place[col]  = numL;
                   numL ++;
                }
                else  // this is not a fill-in element
                {
                   value[pos] -= val; 
                   level[pos] = fsls_min(level[pos],GElevel + levelLU[j] + 1);
                }
             
             }
             else  // dealing with upper part
             {
                if (pos < 0)  // this is a fill-in element
                {
                   if (numU >= n)
                   {
                      jx_printf("\n >> Incomprehensible error, matrix must be wrong!\n\n");
                      ierr = -1;
                      return ierr;
                   }
                   upos = row + numU;
                   value[upos] = -val;
                   index[upos] = col;
                   level[upos] = GElevel + levelLU[j] + 1;
                   place[col]  = upos;
                   numU ++;
                }
                else  // this is not a fill-in element
                {
                   value[pos] -= val;
                   level[pos] = fsls_min(level[pos],GElevel + levelLU[j] + 1); 
                }
             }
          }
         
          GE ++;
          
      } // end while loop
   
   
     /*-------------------------------------------------------
      *  Step 3:  Zero out place[].  (U-part) 
      *------------------------------------------------------*/          
          
      for (j = 0; j < numU; j ++)
      {
         place[index[row+j]] = -1;
      }


     /*-------------------------------------------------------
      *  Step 4:  Update the current row of L and U matrix.
      *------------------------------------------------------*/ 
   
      /* get the L part */
      for (j = 0; j < numL; j ++)
      {
         if (level[j] <= p)
         {
            valueLU[cntmsr] = value[j];
            indexLU[cntmsr] = index[j];
            cntmsr ++;
            if (cntmsr > sizeMSR)
            {
               jx_printf("\nInsufficient storage in L!\n\n");
               ierr = -2;
               return ierr;
            }
         }
      }
      
      /* record the address where the U-part starts in valueLU
         (indexLU or levelLU) in the current row */
      Uhead[row] = cntmsr;
      
      /* get the U part */
      for (j = row + 1; j < row + numU; j ++)
      {
         if (level[j] <= p)
         {
            valueLU[cntmsr] = value[j];
            indexLU[cntmsr] = index[j];
            levelLU[cntmsr] = level[j];
            cntmsr ++;
            if (cntmsr > sizeMSR)
            {
               jx_printf("\n >> Insufficient storage in U!\n\n");
               ierr = -3; return ierr;
            }
         }
      } 
      
      /* get the diagonal element */
      if (value[row] == 0.0)
      {
          jx_printf("\n >> Zero diagonal in Row %d!\n\n",row);
          ierr = -5; return ierr;
      }
      
      valueLU[row] = value[row];
      
      /* update the indexLU[] */
      indexLU[row+1] = cntmsr; 
      
   }  // end the main loop
   
   
   //--------------------------------------------------------------
   //  For the convinience of soling the triangular linear system
   //--------------------------------------------------------------
   
   for (j = 0; j < n; j ++)
   {
      valueLU[j] = 1.0 / valueLU[j];
   }


   //---------------------------------------------
   //  Free some staff and return
   //---------------------------------------------
   
   fsls_TFree(work); 
   fsls_TFree(value);
   *valueLU_ptr = valueLU; 
   *indexLU_ptr = indexLU;   
  
   return ierr;
}
 
/*!
 * \fn JX_Int fsls_ILUp_DecompTest
 * \brief ILU(p) decomposition of A.
 * \param *A pointer to the matrix to be ILU(p)-decomposed
 * \param *p the "level of fill-in" parameter
 * \param **L JX_Real pointer to the lower part of the resulting decomposition
 * \param **U JX_Real pointer to the upper part of the resulting decomposition
 * \return ierr
 *   ierr =  0  --> successful return.
 *   ierr = -1  --> Error,the input matrix may be wrong because the elimination process
 *                  has generated a row in L or U whose length is greater than n. 
 *   ierr = -2  --> The matrix L overflows the array 'valueLU'.
 *   ierr = -3  --> The matrix U overflows the array 'valueLU'.
 *   ierr = -4  --> Illegal value for 'p'.
 *   ierr = -5  --> Zero diagonal in the input matrix.
 * \note (1) All the diagonal elements of the input matrix A must be nonzero.
 *       (2) For better efficiency, you'd better remove all zeros in the matrix A by using  
 *           the function 'fsls_CSRMatrixDeleteZeros(A,0.0)' previously, otherwise, the statement
 *           'if (val == 0.0) continue;' in the function is quite necessary.
 *       (3) In order to save memory, 'sizeMSR' is predicted at the very beginning by predicting 
 *           the number of possible fill-in's during the decomposition.
 * \note if the zeros in A are not removed previously, the output information 
 *          "real number of fill-in's in ILU" may be wrong(say, negative). 2011/03/29
 * \author peghoty
 * \date 2010/03/12
 */
JX_Int
fsls_ILUp_DecompTest( fsls_CSRMatrix   *A, 
                      JX_Int               p,
                      fsls_CSRMatrix  **L_ptr,
                      fsls_CSRMatrix  **U_ptr  )
{
   /* information of the matrix */
   JX_Int     n  = fsls_CSRMatrixNumRows(A);
   JX_Int     nz = fsls_CSRMatrixNumNonzeros(A);
   JX_Int    *ia = fsls_CSRMatrixI(A);
   JX_Int    *ja = fsls_CSRMatrixJ(A);
   JX_Real *a  = fsls_CSRMatrixData(A);
   
   /* storage for L and U in MSR format */
   JX_Int     fillins;
   JX_Int     sizeMSR;
   JX_Real *valueLU = NULL;
   JX_Int    *indexLU = NULL;
   
  /*-----------------------------------------------------------------------------
   *  Auxiliary arrays
   *
   *  1. For the current row, the following local auxiliary arrays are allocated:
   *
   *  (1) value[i],i=0(1)n-1: be used to store the values of all nonzeros 
   *                          in the current row.
   *  (2) index[i],i=0(1)n-1: be used to store the column numbers of all 
   *                          nonzeros in the current row.
   *  (3) level[i],i=0(1)n-1: be used to store the level of fill-in values
   *                          of all nonzeros in the current row.
   *  (4) place[i],i=0(1)n-1: the position of a_{*,i} in 'value' or 'index'.
   *                          
   *  Attention: the above arrays are used in the compressing way, they 
   *             are divided into the Left-part(Lower-part) and Right-part
   *             (Upper-part) by the position 'k', i.e., the row number of 
   *             the current row.
   *
   *  2. Another global auxiliary arrays:
   * 
   *  (1) Uhead[i],i=0(1)n-1: be used to record the address where the U-part
   *                          starts in valueLU (indexLU or levelLU) in row i.                       
   *  (2) levelLU[i],i=0(1)sizeMSR-1: be used to store the level of fill-in 
   *                                  values for all the U-part elements in
   *                                  'valueLU'. (there apparently exists 
   *                                  some waste!!)
   *-----------------------------------------------------------------------------*/

   JX_Real *value   = NULL; 
   JX_Int    *index   = NULL;    
   JX_Int    *place   = NULL;    
   JX_Int    *level   = NULL;    
   JX_Int    *Uhead   = NULL;
   JX_Int    *levelLU = NULL;
   JX_Int    *work    = NULL;
   
   /* local variables */
   JX_Int  i,j,k,ierr = 0;
   JX_Int  numL,numU;    // for counting the nonzeros in the L- and U-part of the current row, respectively
   JX_Int  GE;           // for counting the rows that contribute in the GE of the current row
   JX_Int  mincol,minpos;
   JX_Int  tmpindex,tmplevel;
   JX_Int  row,col,pos,upos,cntmsr;   
   JX_Int  GElevel;
   JX_Real val;
   JX_Real tmpvalue;
   JX_Real factor;   // elimination factor
   
   /* the CSR of L and U */
   fsls_CSRMatrix *L = NULL;
   fsls_CSRMatrix *U = NULL; 
   JX_Int     nzL,nzU;
   JX_Int    *iaL = NULL;
   JX_Int    *jaL = NULL;
   JX_Real *aL  = NULL;
   JX_Int    *iaU = NULL;
   JX_Int    *jaU = NULL;
   JX_Real *aU  = NULL;   
   

   //---------------------------------------------
   //  Is the parameter 'p' legal?
   //---------------------------------------------
   
   if (p < 0)
   {
      jx_printf("\n >> Illegal value for p!\n\n");
      ierr = -4; return ierr;
   }


   //---------------------------------------------
   //  Compute the sizeMSR
   //---------------------------------------------
    
   fillins = fsls_ILUp_FillInPredict(A, p);
   sizeMSR = nz + fillins + 1;
   

   //---------------------------------------------
   //  Allocate memories
   //---------------------------------------------
   
   work  = fsls_CTAlloc(JX_Int, sizeMSR + 4*n);
   index = work;      
   place = index + n; level   = place + n;
   Uhead = level + n; levelLU = Uhead + n;
   
   value = fsls_CTAlloc(JX_Real, n);
   valueLU = fsls_CTAlloc(JX_Real, sizeMSR);
   indexLU = fsls_CTAlloc(JX_Int, sizeMSR);
   
  
   //---------------------------------------------
   //  Initialize 'place', cntmsr and indexLU[0]
   //---------------------------------------------
        
   for (j = 0; j < n; j ++) place[j] = -1;
   cntmsr = n + 1;
   indexLU[0] = cntmsr;


   //---------------------------------------------
   //  The main loop of ILU(p) decomposition
   //---------------------------------------------
      
   for (row = 0; row < n; row ++)
   {
   
     /*-------------------------------------------------------
      *  Step 1:  Assign the nonzeros of the current row into
      *  L-part and U-part,respectively, and generate all the
      *  working arrays: value,index,level,place.
      *------------------------------------------------------*/ 
                
      numL = 0;  
      numU = 1;  

      for (j = ia[row]; j < ia[row+1]; j ++)
      {
         val = a[j];   // value
         col = ja[j];  // column number
         
         if (val == 0.0) continue;
      
         if (col < row)
         {
            value[numL] = val;
            index[numL] = col;
            level[numL] = 0;
            place[col]  = numL;
            numL ++; 
         }
         else if (col == row)
         {
            value[row] = val;
            index[row] = col;
            level[row] = 0;
            place[col] = row;
         }
         else
         {
            pos = row + numU;
            value[pos] = val;
            index[pos] = col;
            level[pos] = 0;
            place[col] = pos;
            numU ++;
         } 
      }  
  
   
     /*-------------------------------------------------------
      *  Step 2:  Eliminate previous rows.
      *------------------------------------------------------*/ 
      
      GE = 0;
      
      while (GE < numL)
      {

         /*-------------------------------------------------------
          *  Step 2.1  In order to do the elimination in the   
          *  correct order we must select the smallest column
          *  index among index[k], k=GE,...,numL-1.
          *------------------------------------------------------*/
          
          mincol = index[GE];
          minpos = GE;
          
          for (j = GE+1; j < numL; j ++)
          {
             if (index[j] < mincol)
             {
                mincol = index[j];
                minpos = j;
             }
          }
          
          
         /*-------------------------------------------------------
          *  Step 2.2  if (minpos != GE), do the exchanging in
          *  the following arrays: value,index,place,level.
          *------------------------------------------------------*/
          
          if (minpos != GE)
          {
             /* swap in "index" */
             tmpindex = index[GE];
             index[GE] = index[minpos];
             index[minpos] = tmpindex;
             
             /* swap in "place" */
             place[mincol]   = GE;
             place[tmpindex] = minpos;
             
             /* swap in "level" */
             tmplevel = level[GE];
             level[GE] = level[minpos];
             level[minpos] = tmplevel;

             /* swap in "value" */
             tmpvalue = value[GE];
             value[GE] = value[minpos];
             value[minpos] = tmpvalue;             
          } 


         /*-------------------------------------------------------
          *  Step 2.3  Zero out place[mincol].  (L-part) 
          *------------------------------------------------------*/
          place[mincol] = -1;
          
          
         /*----------------------------------------------------------------
          *  Step 2.4  Eliminate the current row using the mincol-th row.
          *---------------------------------------------------------------*/   
          
          /* get the multiplier and the corresponding level of fill-in */
          factor  = value[GE] / valueLU[mincol];
          GElevel = level[GE];
          
          if (GElevel > p)
          {
             GE ++;
             continue;
          }
          
          /* update the current row */
          value[GE] = factor;
          for (j = Uhead[mincol]; j < indexLU[mincol+1]; j ++)
          {
             val = factor*valueLU[j];
             col = indexLU[j];
             pos = place[col];
             
             if (col < row)      // dealing with lower part
             {
                if (pos < 0)     // this is a fill-in element
                {
                   if (numL >= n)
                   {
                      jx_printf("\n >> Incomprehensible error, matrix must be wrong!\n\n");
                      ierr = -1;
                      return ierr;
                   }
                   value[numL] = -val;
                   index[numL] = col;
                   level[numL] = GElevel + levelLU[j] + 1;
                   place[col]  = numL;
                   numL ++;
                }
                else  // this is not a fill-in element
                {
                   value[pos] -= val; 
                   level[pos] = fsls_min(level[pos],GElevel + levelLU[j] + 1);
                }
             
             }
             else  // dealing with upper part
             {
                if (pos < 0)  // this is a fill-in element
                {
                   if (numU >= n)
                   {
                      jx_printf("\n >> Incomprehensible error, matrix must be wrong!\n\n");
                      ierr = -1;
                      return ierr;
                   }
                   upos = row + numU;
                   value[upos] = -val;
                   index[upos] = col;
                   level[upos] = GElevel + levelLU[j] + 1;
                   place[col]  = upos;
                   numU ++;
                }
                else  // this is not a fill-in element
                {
                   value[pos] -= val;
                   level[pos] = fsls_min(level[pos],GElevel + levelLU[j] + 1); 
                }
             }
          }
         
          GE ++;
          
      } // end while loop
   
   
     /*-------------------------------------------------------
      *  Step 3:  Zero out place[].  (U-part) 
      *------------------------------------------------------*/          
          
      for (j = 0; j < numU; j ++)
      {
         place[index[row+j]] = -1;
      }


     /*-------------------------------------------------------
      *  Step 4:  Update the current row of L and U matrix.
      *------------------------------------------------------*/ 
   
      /* get the L part */
      for (j = 0; j < numL; j ++)
      {
         if (level[j] <= p)
         {
            valueLU[cntmsr] = value[j];
            indexLU[cntmsr] = index[j];
            cntmsr ++;
            if (cntmsr > sizeMSR)
            {
               jx_printf("\nInsufficient storage in L!\n\n");
               ierr = -2;
               return ierr;
            }
         }
      }
      
      /* record the address where the U-part starts in valueLU
         (indexLU or levelLU) in the current row */
      Uhead[row] = cntmsr;
      
      /* get the U part */
      for (j = row + 1; j < row + numU; j ++)
      {
         if (level[j] <= p)
         {
            valueLU[cntmsr] = value[j];
            indexLU[cntmsr] = index[j];
            levelLU[cntmsr] = level[j];
            cntmsr ++;
            if (cntmsr > sizeMSR)
            {
               jx_printf("\n >> Insufficient storage in U!\n\n");
               ierr = -3; return ierr;
            }
         }
      } 
      
      /* get the diagonal element */
      if (value[row] == 0.0)
      {
          jx_printf("\n >> Zero diagonal in Row %d!\n\n",row);
          ierr = -5; return ierr;
      }
      
      valueLU[row] = value[row];
      
      /* update the indexLU[] */
      indexLU[row+1] = cntmsr; 
      
   }  // end the main loop

   jx_printf("\n predicted number of fill-in's in ILU(%d) = %d", p, fillins);
   jx_printf("\n real      number of fill-in's in ILU(%d) = %d\n", p, cntmsr-nz-1);


   //---------------------------------------------
   //  Free some staff
   //---------------------------------------------
   
   fsls_TFree(work); fsls_TFree(value);

   //----------------------------------------------------------------
   //  Split L and U from the arrays 'valueLU' and 'indexLU' (MSR)
   //----------------------------------------------------------------
   
   /* compute the nz's for L and U */
   nzL = n; nzU = n;
   for (i = 0; i < n; i++)
   {
      for (j = indexLU[i]; j < indexLU[i+1]; j ++)
      {
         k = indexLU[j]; // column number
         if (k < i)
           nzL ++;
         else
           nzU ++;
      }
   }
 
   /* Create L */
   L = fsls_CSRMatrixCreate(n,n,nzL);
   fsls_CSRMatrixInitialize(L);
   iaL = fsls_CSRMatrixI(L);
   jaL = fsls_CSRMatrixJ(L);
   aL  = fsls_CSRMatrixData(L);

   /* Create U */
   U = fsls_CSRMatrixCreate(n,n,nzU);
   fsls_CSRMatrixInitialize(U);
   iaU = fsls_CSRMatrixI(U);
   jaU = fsls_CSRMatrixJ(U);
   aU  = fsls_CSRMatrixData(U);
   
   /* Generate L and U */
   nzL = 0; nzU = 0;   
   for (i = 0; i < n; i ++)
   {
      iaL[i] = nzL;
      iaU[i] = nzU;
      
      jaL[nzL] = i;
       aL[nzL] = 1.0;
      jaU[nzU] = i;
       aU[nzU] = valueLU[i];
       
      nzL ++; nzU ++;
      
      for (j = indexLU[i]; j < indexLU[i+1]; j ++)
      {
         k = indexLU[j]; // column number
         if (k < i)
         {
           jaL[nzL] = k;
            aL[nzL] = valueLU[j];
           nzL ++;
         }
         else
         {
           jaU[nzU] = k;
            aU[nzU] = valueLU[j];
           nzU ++;
         }
      }
   }   
   iaL[n] = nzL; iaU[n] = nzU;
   
   fsls_TFree(indexLU); fsls_TFree(valueLU);
   *L_ptr = L; *U_ptr = U;  
  
   return ierr;
}
