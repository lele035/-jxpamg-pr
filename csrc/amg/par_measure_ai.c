//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_splitting_ai.c
 *  Date: 2013/01/21
 */ 

#include "jx_pamg.h"

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

/*!
 * \fn JX_Int jx_PAMGMeasureAI
 * \brief AI Splitting routine.
 * measure_ai for point i:
 *
 *    > 0.0: ai-points
 *   =< 0.0: not ai-points
 *      (1) = 0.0: not ai-points, normal points.
 *      (2) =-1.0: not ai-points, parasitic points.
 *      (3) =-2.0: not ai-points, diagonal-dominant pionts.
 *                 including source points and isolated points.
 *      (4) other: not ai-points.
 * \date 2013/03/10
 */
JX_Int
jx_PAMGMeasureAI( jx_ParCSRMatrix  *par_S,
                  jx_ParCSRMatrix  *par_A,
                  JX_Int               debug_flag,
                  JX_Real          **measure_ai_ptr )
{
   MPI_Comm               comm        = jx_ParCSRMatrixComm(par_S);
   jx_ParCSRCommPkg      *comm_pkg    = jx_ParCSRMatrixCommPkg(par_S);
   jx_ParCSRCommHandle   *comm_handle = NULL;

   jx_CSRMatrix    *S_diag    = jx_ParCSRMatrixDiag(par_S);
   JX_Int             *S_diag_i  = jx_CSRMatrixI(S_diag);
   JX_Int             *S_diag_j  = jx_CSRMatrixJ(S_diag);

   jx_CSRMatrix    *S_offd    = jx_ParCSRMatrixOffd(par_S);
   JX_Int             *S_offd_i  = jx_CSRMatrixI(S_offd);
   JX_Int             *S_offd_j  = NULL;

   JX_Int              num_variables = jx_CSRMatrixNumRows(S_diag);
   JX_Int 		    num_cols_offd = 0;

   JX_Int		    num_sends = 0;
   JX_Int  	   *int_buf_data;
   JX_Real	   *buf_data;
                      
   JX_Real          *measure_ai;
   JX_Real          *measure_inf;
                      
   JX_Int              i, j;
   JX_Int              index, start, my_id, num_procs;
                      
   JX_Int              ierr = 0;

   jx_ParCSRMatrix  *par_ST;
   jx_CSRMatrix    *ST_diag;
   JX_Int             *ST_diag_i;
   JX_Int             *ST_diag_j;
   jx_CSRMatrix    *ST_offd;
   JX_Int             *ST_offd_i;
   JX_Int             *ST_offd_j;


   JX_Int nzi, nzit, jt;
   JX_Int nodep;
   JX_Int measure_dep, measure_dep_1;
   JX_Int *measure_dep_2; // number of neighbors for strong dependence each other.
//   JX_Real mai_1, mai_2;

   JX_Int ii;
   JX_Real buf_data_tmp;

  /***********************************************************************
   *  BEFORE THE INDEPENDENT SET COARSENING LOOP:
   *   measure_inf: calculate the measures, and communicate them
   *     (this array contains measures for both local and external nodes)
   *   CF_marker, CF_marker_offd: initialize CF_marker
   *     (separate arrays for local and external; 
   *      0=unassigned, negative=F point, positive=C point)
   ***********************************************************************/      

  /*-------------------------------------------------------------------------
   * Use the ParCSR strength matrix, par_S.
   *
   * For now, the "strength" of dependence/influence is defined in
   * the following way: i depends on j if
   *     aij > jx_max (k != i) aik,    aii < 0
   * or
   *     aij < jx_min (k != i) aik,    aii >= 0
   * Then S_ij = 1, else S_ij = 0.
   *
   * NOTE: S_data is not used; in stead, only strong columns are retained
   *       in S_j, which can then be used like S_data
   *-----------------------------------------------------------------------*/

   jx_MPI_Comm_size(comm,&num_procs);
   jx_MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_A);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
   buf_data     = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
 
   num_cols_offd = jx_CSRMatrixNumCols(S_offd);

   S_diag_j = jx_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = jx_CSRMatrixJ(S_offd);
   }

   jx_ParCSRMatrixTranspose( par_S,
                             &par_ST,
                             0 );

   ST_diag    = jx_ParCSRMatrixDiag(par_ST);
   ST_diag_i  = jx_CSRMatrixI(ST_diag);
   ST_diag_j  = jx_CSRMatrixJ(ST_diag);

   ST_offd    = jx_ParCSRMatrixOffd(par_ST);
   ST_offd_i  = jx_CSRMatrixI(ST_offd);
   ST_offd_j  = jx_CSRMatrixJ(ST_offd);

  /*---------------------------------------------------------------
   * Compute the measures
   *
   * The measures are currently given by the column sums of par_S.
   * Hence, measure_inf[i] is the number of influences
   * of variable i.
   *
   * The measures are augmented by a random number
   * between 0 and 1.
   *-------------------------------------------------------------*/

   measure_dep_2 = jx_CTAlloc(JX_Int, num_variables);
   measure_ai = jx_CTAlloc(JX_Real, num_variables + num_cols_offd);
   measure_inf = jx_CTAlloc(JX_Real, num_variables + num_cols_offd);

   /* first calculate the local part of the sums for the external nodes */
   for (i = 0; i < S_offd_i[num_variables]; i ++)
   { 
      measure_ai[num_variables + S_offd_j[i]] += 1.0;
      measure_inf[num_variables + S_offd_j[i]] += 1.0;
   }

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   /* NOTE BY XU: for measure_ai, also should be considered parallel implementation. */
   if (num_procs > 1)
   {
      comm_handle = jx_ParCSRCommHandleCreate(2, comm_pkg, &measure_inf[num_variables], buf_data);
   }

   /* calculate the local part for the local nodes */
   for (i = 0; i < S_diag_i[num_variables]; i ++)
   {
       measure_inf[S_diag_j[i]] += 1.0;
   }

   for (i = 0; i < num_variables; i ++)
   {
       measure_ai[i] = 0.0;
   }

#if 0 
   /* calculate the local part for the local nodes */
   for (i = 0; i < num_variables; i ++) {
      for (nzi = S_diag_i[i]; nzi < S_diag_i[i+1]; nzi ++) {
          j = S_diag_j[nzi];
          nodep = 1;
          /* determine if the j depend on i */
          for (nzj = S_diag_i[j]; nzj < S_diag_i[j+1]; nzj ++) {
              if (S_diag_j[nzj] == i) {
                 nodep = 0;
                 break;
              }
          }
          if (nodep == 1) {
              measure_ai[j] += 1.0;
          }
      }
   }
#endif

#if 1 
   /* calculate the local and neighbor parts using par_ST. */
   for (i = 0; i < num_variables; i ++) {
      /* diagonal part. */
      for (nzi = S_diag_i[i]; nzi < S_diag_i[i+1]; nzi ++) {
          j = S_diag_j[nzi];
          nodep = 1;
          /* determine if the j depend on i */
          for (nzit = ST_diag_i[i]; nzit < ST_diag_i[i+1]; nzit ++) {
             jt = ST_diag_j[nzit];
             if (jt == j) {
                nodep = 0;
                measure_dep_2[i]++;
                //break;
             }
          }
          if (nodep == 1) {
              measure_ai[j] += 1.0;
          }
      }

      /* off-diagonal part. */
      for (nzi = S_offd_i[i]; nzi < S_offd_i[i+1]; nzi ++) {
          j = S_offd_j[nzi];
          nodep = 1;
          /* determine if the j depend on i */
          for (nzit = ST_offd_i[i]; nzit < ST_offd_i[i+1]; nzit ++) {
             jt = ST_offd_j[nzit];
             if (jt == j) {
                nodep = 0;
                measure_dep_2[i]++;
                //break;
             }
          }
          if (nodep == 1) {
              measure_ai[j] += 1.0;
          }
      }
   }
#endif

   /* finish the communication */
   if (num_procs > 1)
   {
      jx_ParCSRCommHandleDestroy(comm_handle);
   }
       
   /* need checked if num_procs. */
   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
      {
         ii = jx_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         buf_data_tmp = buf_data[index++];
         measure_inf[ii] += buf_data_tmp;
      }
   }

   /* set the measures of the external nodes to zero */
   for (i = num_variables; i < num_variables + num_cols_offd; i ++)
   { 
      measure_ai[i] = 0.0;
      measure_inf[i] = 0.0;
   }

   /* normalizing measure_ai */
   if (debug_flag == 4) 
   {
      jx_printf(" measure_ai: \n");
   }
   for (i = 0; i < num_variables; i ++)
   { 
      measure_dep = S_diag_i[i+1]-S_diag_i[i];
      measure_dep = measure_dep + S_offd_i[i+1]-S_offd_i[i];

      measure_dep_1 = measure_dep - measure_dep_2[i];
      if (measure_dep == 0) {
         if (measure_dep_1 != 0 || measure_dep_2[i] !=0 ) {
            jx_printf("for point %d : \n", i);
            jx_printf("measure_dep == 0, however, measure_dep_1 != 0 || measure_dep_2 !=0 \n");
            exit(0);
         }
      }
 
      if (measure_ai[i] > 0.0 && measure_dep >0) 
      {
#if 1 
         if (measure_dep_1 == 0) {
            measure_ai[i] = measure_ai[i]/(measure_dep+measure_ai[i]) + 1.0;
         } else {
            measure_ai[i] = measure_ai[i]/(measure_dep+measure_ai[i]);
         }
#endif

         //measure_ai[i] = measure_ai[i]/(measure_dep+measure_ai[i]);

#if 0         
         mai_1 = measure_ai[i]/(measure_ai[i]+measure_dep);
         mai_2 = measure_ai[i]/(measure_ai[i]+measure_dep_1);
         measure_ai[i] = mai_1 + mai_2;
#endif
      } 
      else if (measure_dep > 0) 
      {
         measure_ai[i] = (measure_inf[i]-measure_dep)/measure_dep;
      } 
      else // diagonal-dominant points: source point and isolate points. 
      {
         measure_ai[i] = -2.0;
      }

      if (debug_flag == 4) 
      {
         //if ( measure_ai[i] > 1.0e-10 || measure_ai[i] < -1.0e-10) 
         //{ 
            jx_printf(" i= %d, %f \n", i, measure_ai[i]);
         //}
      }
  
   }

   jx_TFree(int_buf_data);
   jx_TFree(buf_data);
   jx_TFree(measure_dep_2);
   jx_TFree(measure_inf);
   jx_ParCSRMatrixDestroy(par_ST);

   *measure_ai_ptr = measure_ai;

   return (ierr);
}
