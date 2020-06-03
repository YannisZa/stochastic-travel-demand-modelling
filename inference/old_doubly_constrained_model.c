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



// Computes flow estimates using the DSF procedure - see page 373 of "Gravity Models of Spatial Interaction Behavior" book.
void infer_flows_dsf_procedure(const int *orig_supply,
                                const int *dest_demand,
                                const double* cost_mat,
                                const size_t n,
                                const size_t m,
                                const double beta,
                                double* flows,
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

    for(i=0; i<n; i++) {

        for(j=0; j<m; j++) {
          // Initialise flows
          // See equation (5.61) page 373 from Gravity Models of Spatial Interaction Behavior book
          // Assume Q_j = P_i = 1 for all i,j
          flows[i*m + j] = exp(-beta*cost_mat[i*m + j]);
        }
    }

    // Solve normalising factors iteratively
    t = 0;
    // while (fabs(inferred_total_flow-total_flow) > 0.001 || t < max_iters) {
    while (t < max_iters) {
        // Loop over origins and reset total flow inferred
        // double inferred_total_flow = 0;
        for(i=0; i<n; i++) {
            // Loop over destinations
            for(j=0; j<m; j++) {
                // To avoid creating (max_iters)x(N) and (max_iters)x(M) A and B matrices
                // that overload memmory, 1xN and 1xM A and B matrices are created, respectively.
                // Size 1 comes from the fact that the depth of recursion is 1

                if (t % 2 == 1) {
                  // NOTE: A_i is NOT CURRENTLY USED ANYWHERE BUT MAY BE IN THE FUTURE
                  // Update $A_i$
                  // $A_i = 1 / \exp(-1) \sum_{p=0}^M B_j D_j \exp(-\beta c_{pj})$
                  // A_i^(2r-1) = O_i / (\sum_{j=1}^M B_j^(2r-2) F_{ij})
                  // See page 374 from Gravity Models of Spatial Interaction Behavior book

                  /*
                  temp = 0;
                  for(jj=0; jj<m; jj++) {
                    temp += B_vec[jj] * exp(-beta * cost_mat[i*m + jj]);
                  }
                  temp = 1. / temp;
                  A_vec[i] = orig_supply[i] * temp;
                  */

                  // Add solution for flows: $T_{ij} = A_i B_j O_i D_j \exp(-\beta c_{ij})$
                  // by casting result to integer
                  // T_{ij}^(2r-1) = T_{ij}^(2r-2) O_i / T_{i+}^{2r-2}
                  // See equation (5.62) page 374 from Gravity Models of Spatial Interaction Behavior book
                  temp = 0;
                  for(jj=0; jj<m; jj++) {
                    temp += flows[i*m + jj];
                  }
                  temp = 1. / temp;
                  flows[i*m + j] = flows[i*m + j] * orig_supply[i] * temp;


                } else {
                  // NOTE: B_J is NOT CURRENTLY USED ANYWHERE BUT MAY BE IN THE FUTURE
                  // Update $B_j$
                  // $B_j = 1 / \exp(-1) \sum_{l=0}^N A_i O_i \exp(-\beta c_{lj})$
                  // B_j^(2r) = D_j / (\sum_{i=1}^N A_i^(2r-1) F_{ij})
                  // See page 374 from Gravity Models of Spatial Interaction Behavior book

                  /*
                  temp = 0;
                  for (ii=0; ii<n; ii++) {
                      temp += A_vec[ii] * exp(-beta * cost_mat[ii*m + j]);
                  }
                  temp = 1. / temp;
                  B_vec[j] = dest_demand[j] * temp;
                  */

                  // Add solution for flows: $T_{ij} = A_i B_j O_i D_j \exp(-\beta c_{ij})$
                  // by casting result to integer
                  // T_{ij}^(2r) = T_{ij}^(2r-1) D_j / T_{+j}^{2r-1}
                  // See equation (5.63) page 374 from Gravity Models of Spatial Interaction Behavior book
                  temp = 0;
                  for(ii=0; ii<n; ii++) {
                    temp += flows[ii*m + j];
                  }
                  temp = 1. / temp;
                  flows[i*m + j] = flows[i*m + j] * dest_demand[j] * temp;
                }
                // See equation (5.66) page 374 from Gravity Models of Spatial Interaction Behavior book
                /* flows[i*m + j] = A_vec[i] * B_vec[j] * exp(-beta * cost_mat[i*m + j]); */
                // Add flow to total_flow
                // inferred_total_flow += flows[i*m + j];

                // Print statements
                if (show_params == 1) {
                    printf("A:  ");
                    for(ii=0; ii<n; ii++) {
                      printf("%f",A_vec[ii]);
                    }
                    printf("\n");
                    printf("B:  ");
                    for(jj=0; jj<m; jj++) {
                      printf("%f",B_vec[jj]);
                    }
                    printf("\n");
                  }

                  // Print statements
                  if (show_flows == 1) {
                      printf("Flow");
                      printf("[");
                      printf("%zu",i);
                      printf(",");
                      printf("%zu",j);
                      printf("]:  ");
                      printf("%f",flows[i*m + j]);
                      printf("\n");
                  }
            }

      }
      // Increment number of iterations
      t += 1;

  }
}
