//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGRelaxIF
 * \brief This function was ignored before, because I took it as granted that 
 *        the array (or pointer) "grid_relax_points", as one member of the struct
 *        "jx_ParAMGData" must be set, however, it's not necessary.
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelaxIF( jx_ParCSRMatrix *par_matrix,
                jx_ParVector    *par_rhs,
                JX_Int             *cf_marker,
                JX_Int              relax_type,
                JX_Int              relax_order,
                JX_Int              cycle_type,
                JX_Real           relax_weight,
                JX_Real           omega,
                jx_ParVector    *par_app,
                jx_ParVector    *Vtemp )
{
   JX_Int i, Solve_err_flag = 0;
   JX_Int relax_points[2];

   if (relax_order == 1 && cycle_type < 3)
   {
       if (cycle_type < 2)
       {
          relax_points[0] =  1;
          relax_points[1] = -1;
       }
       else
       {
	      relax_points[0] = -1;
	      relax_points[1] =  1;
       }

       for (i = 0; i < 2; i ++)
       {
         //  printf("前后磨光\n");
          Solve_err_flag = jx_PAMGRelax( par_matrix,
                                         par_rhs,
                                         cf_marker,
                                         relax_type,
                                         relax_points[i],
                                         relax_weight,
                                         omega,
                                         par_app,
                                         Vtemp ); 
       }
   }
   else
   {
#if 0
      //  for (i = 0; i < 5; i ++)
       Solve_err_flag = jx_PAMGRelax( par_matrix,
                                      par_rhs,
                                      cf_marker,
                                      relax_type,
                                      0,
                                      relax_weight,
                                      omega,
                                      par_app,
                                      Vtemp ); 
#else
      // 粗空间解法器，通过tol来控制, zhaoli.2021.07.09
      // printf("粗空间吗？relax_type %d\n", relax_type);
      int isPrint = 0, covstd = 2; // 1: residual, 2: RRN
      int maxit = 50;
      double TOL = 1e-2;
      double r0_norm2, r_norm2, r_norm2_old, RRN;
      jx_ParVector * R;
      R = jx_ParVectorCreate(jx_ParVectorComm(par_rhs), jx_ParVectorGlobalSize(par_rhs), jx_ParVectorPartitioning(par_rhs));
      jx_ParVectorInitialize(R);
      jx_ParVectorCopy(par_rhs, R);
      // y := alpha*A*x + beta*y.
      // R = b - Ax
      jx_ParCSRMatrixMatvec(-1, par_matrix, par_app, 1, R);
      r0_norm2 = jx_ParVectorInnerProd(R, R);
      r0_norm2 = sqrt(r0_norm2);
      if (isPrint)
      {
         printf("\nCoarsest Space, matrix size = %d, relax_type = %d:\n", par_matrix->global_num_rows, relax_type);
         if (covstd == 1)
         {
            /* pure absolute tolerance */
            jx_printf("Iters       ||r||_2     conv.rate\n");
            jx_printf("-----    ------------   ---------\n");
         }else if (covstd == 2)
         {
            jx_printf("Iters       ||r||_2     conv.rate  ||r||_2/||b||_2\n");
            jx_printf("-----    ------------   ---------  ------------ \n");
         }else{

         }
         jx_printf("% 5d    %e\n", 0, r0_norm2);
      }

      r_norm2 = r0_norm2;
      for (i = 0; i < maxit; i ++)
      {
         r_norm2_old = r_norm2;
         Solve_err_flag = jx_PAMGRelax( par_matrix,
                                       par_rhs,
                                       cf_marker,
                                       relax_type,
                                       0,
                                       relax_weight,
                                       omega,
                                       par_app,
                                       Vtemp );
         jx_ParVectorCopy(par_rhs, R);                                       
         jx_ParCSRMatrixMatvec(-1, par_matrix, par_app, 1, R);
         r_norm2 = jx_ParVectorInnerProd(R, R);
         r_norm2 = sqrt(r_norm2);

         if(covstd == 1){
               /* pure absolute tolerance */
               if (isPrint) jx_printf("% 5d    %e    %f\n", i+1, r_norm2, r_norm2/r_norm2_old);
         }
         else if (covstd == 2)
         {
            RRN = r_norm2 / r0_norm2;
            if (isPrint) jx_printf("% 5d    %e    %f    %e\n", i+1, r_norm2, r_norm2/r_norm2_old, RRN); 
         }else{

         }
         
         if(covstd == 1){
            if (r_norm2 < TOL) break;   
         }
         else if (covstd == 2)
         {
            if (RRN < TOL) break;   
         }else{
            
         }                                       
      }

      if(isPrint || i == maxit){
         if(covstd == 1){
            printf("粗空间的迭代次数为 %d，设置的 tol = %e, Residual Norm = %e\n", i, TOL, r_norm2);             
         }
         else if (covstd == 2)
         {
            printf("粗空间的迭代次数为 %d，设置的 tol = %e, Relative Residual Norm = %e\n", i, TOL, RRN);             
         }else{
            
         }
      }
#endif                                      
   }

   return Solve_err_flag;
}
