#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/*
TO DO: Create data structure to store and return flows, A,B matrices
 */

//  Data struture for storing results of model
// struct Tuple {
//     int** f;
//     double* A;
//     double* B;
// };
//
// //
// struct Tuple getPair() {
//     Tuple r = { 1, getString() };
//     return r;
// }



void infer_flows(const int *orig_supply,
                const int *dest_demand,
                const double* cost_mat,
                const size_t n,
                const size_t m,
                const double beta,
                int* flows,
                double* A_vec,
                double* B_vec,
                const size_t max_iters,
                const bool show_params,
                const bool show_flows)
{
    size_t i,ii,j,jj,t;
    double temp;

    /*
    orig_supply [array]: supply generated at each origin zone
    dest_demand [array]: demand generated at each destination zone
    cost_mat [array]: matrix of costs travelling from each origin to each destination
    n [int]: number of origin zones
    m [int]: number of destination zones
    beta [double]: parameter of model controlling effect of cost matrix on the flow update
    A_vec [array]: A_i = \exp(-lambda_i)/O_i
    B_vec [array]: B_j = \exp(-kappa_j)/D_j
    max_iters [int]: maximum iterations for which system of equations should be solved
    show_params [boolean]: flag for printing parameter matrices A and B
    show_flows [boolean]: flag for printing flow from each origin to each destination
    */


    // Initialise empty flow matrix
    for(i = 0; i < n; ++i)
        for(j = 0; j < m; ++j)
            flows[i*m + j] = 0;

    for(i=0; i<n; i++) {
        // flows[i]=(int *)malloc(sizeof(int)*n);
        for(j=0; j<m; j++) {
            // Solve normalising factors iteratively
            t = 0;
            while (t < max_iters - 1) {
                // To avoid creating (max_iters)x(N) and (max_iters)x(M) A and B matrices
                // that overload memmory, 1xN and 1xM A and B matrices are created, respectively.
                // Size 1 comes from the fact that the depth of recursion is 1

                // Update $A_i$
                // $A_i = 1 / \exp(-1) \sum_{p=0}^M B_j D_j \exp(-\beta c_{pj})$
                temp = 0;
                for(jj=0; jj<m; jj++) {
                  temp += B_vec[jj] * dest_demand[jj] * exp(-beta * cost_mat[i*m + jj]);
                }
                A_vec[i] = 1. / temp;

                // Update $B_j$
                // $B_j = 1 / \exp(-1) \sum_{l=0}^N A_i O_i \exp(-\beta c_{lj})$
                temp = 0;
                for(ii=0; ii<n; ii++) {
                    temp += A_vec[ii] * orig_supply[ii] * exp(-beta * cost_mat[ii*m + j]);
                }
                B_vec[m] = 1. / temp;

                // Increment number of iterations
                t += 1;
            }
            // Print statement
            if (show_params) {
                printf("A");
                for(ii=0; ii<n; ii++) {
                  printf("%f",A_vec[ii]);
                }
                printf("\n");
                printf("B");
                for(jj=0; jj<m; jj++) {
                  printf("%f",B_vec[jj]);
                }
                printf("\n");
            }
            // Add solution for flows: $T_{ij} = A_i B_j O_i D_j \exp(-\beta c_{ij})$
            // by casting result to integer
            flows[i*m + j] = (int) (A_vec[i]*B_vec[j]*orig_supply[i]*dest_demand[j]*exp(-beta*cost_mat[i*m + j]));

            // Print statement
            if (show_flows) {
                printf("Flow");
                printf("%d",flows[i*m + j]);
                printf("\n");
            }
      }
  }
}
