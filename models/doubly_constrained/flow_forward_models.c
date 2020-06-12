#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>


// Computes flow estimates using the Iterative proportional filtering procedure - see page 373 of "Gravity Models of Spatial Interaction Behavior" book.
double *infer_flows_ipf_procedure(double *flows,
                                const int *orig_supply,
                                const int *dest_demand,
                                const double *cost_mat,
                                const double *w_vec,
                                double *A_vec,
                                double *B_vec,
                                const size_t n,
                                const size_t m,
                                const double *theta,
                                const size_t max_iters,
                                const double tolerance,
                                const bool show_flows) {

    /*
    orig_supply [array]: supply generated at each origin zone
    dest_demand [array]: demand generated at each destination zone
    cost_mat [array]: matrix of costs travelling from each origin to each destination
    w_vec [array]: array of "sizes" (e.g. emissions) at each destination
    n [int]: number of origin zones
    m [int]: number of destination zones
    theta [array]: parameters of model
      theta[0]: alpha - parameter controlling effect of destination sizes on flow update
      theta[1]: beta - parameter controlling effect of cost matrix on the flow update
    max_iters [int]: maximum iterations for which system of equations should be solved
    show_params [boolean]: flag for printing parameter matrices A and B
    show_flows [boolean]: flag for printing flow from each origin to each destination
    */


    size_t i,ii,j,jj;
    unsigned int t;
    double temp,error,dest_error,orig_error;
    double estimated_orig[n];
    double estimated_dest[m];


    // Unpack parameter vector
    double alpha = theta[0];
    double beta = theta[1];

    // Initialise flows
    // See equation (5.61) page 373 from Gravity Models of Spatial Interaction Behavior book
    for(i=0; i<n; i++) {

        for(j=0; j<m; j++) {
          // Assume Q_j = P_i = 1 for all i,j
          flows[i*m + j] = 1; // exp(-beta*cost_mat[i*m + j]);
        }
    }

    // Solve iteratively  'max_iters' times
    dest_error = INFINITY;
    orig_error = INFINITY;

    // Update flows until errors are minimised
    t = 0;
    while ( (dest_error > m*tolerance || orig_error > n*tolerance) & (t < max_iters) ){

        // Loop over origins and reset total flow inferred
        for(i=0; i<n; i++) {
            // Loop over destinations
            for(j=0; j<m; j++) {
                // To avoid creating (max_iters)x(N) and (max_iters)x(M) A and B matrices
                // that overload memmory, 1xN and 1xM A and B matrices are created, respectively.
                // Size 1 comes from the fact that the depth of recursion is 1

                // Update A vector
                // A_{i}^{(2r-1)} &= ( \sum_{j=1}^M D_jW_j^{\alpha}B_j^{(2r)} \exp(-\beta c_{ij}) ) ^ {-1}
                temp = 0.;
                for (jj=0; jj<m; jj++) {
                  temp += pow(w_vec[jj],alpha)*dest_demand[jj]*B_vec[jj]*exp(-beta*cost_mat[i*m + jj]);
                }
                A_vec[i] = 1. / temp;

                // Update B vector
                // B_{j}^{(2r)} &= (W_j^{\alpha} \sum_{i=1}^N O_iA_i^{(2r-1)} \exp(-\beta c_{ij}) )^{-1}
                temp = 0.;
                for (ii=0; ii<n; ii++) {
                  temp += orig_supply[ii]*A_vec[ii]*exp(-beta*cost_mat[ii*m + j]);
                }
                temp = pow(w_vec[j],alpha)*temp;
                B_vec[j] = 1. / temp;


                // Update flows
                // Compute flows T_{ij} = A_i B_j O_i D_j W_j^{\alpha} \exp(-\beta c_{ij})
                flows[i*m + j] = A_vec[i]*B_vec[j]*orig_supply[i]*dest_demand[j]*pow(w_vec[j],alpha)*exp(-beta*cost_mat[i*m + j]);

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

        // Update estimated origin supplies
        for(ii=0; ii<n; ii++) {
          temp = 0;
          for(jj=0; jj<m; jj++) {
            temp += flows[ii*m + jj];
          }
          // printf("------------------------\n");
          // printf("Estimated origin[");
          // printf("%zu]: ",ii);
          // printf("%f\n",temp);

          estimated_orig[ii] = temp;
        }
        // Update estimated destination demands
        for(jj=0; jj<m; jj++) {
          temp = 0;
          for(ii=0; ii<n; ii++) {
            temp += flows[ii*m + jj];
          }
          // printf("------------------------\n");
          // printf("Estimated demand[");
          // printf("%zu]: ",jj);
          // printf("%f\n",temp);

          estimated_dest[jj] = temp;
        }
        // Compute total error E = \sum_{i=1}^N abs(O_i-\hat{O_i}) + \sum_{j=1}^M abs(D_j-\hat{D_j})
        error = 0;
        dest_error = 0;
        orig_error = 0;
        for(ii=0; ii<n; ii++) {
            // Compute origin error \sum_{i=1}^N abs(O_i-\hat{O_i})
            orig_error += fabs(estimated_orig[ii]-orig_supply[ii]);
        }
        for(jj=0; jj<m; jj++) {
            // Compute destination error \sum_{i=1}^N abs(O_i-\hat{O_i})
            dest_error += fabs(estimated_dest[jj]-dest_demand[jj]);
        }
        // Total error
        error = dest_error + orig_error;

        printf("------------------------\n");
        printf("Origin error: ");
        printf("%f\n",orig_error);
        printf("Destination error: ");
        printf("%f\n",dest_error);
        printf("Total error: ");
        printf("%f\n",error);

        // Increment number of iterations
        t++;

    }
    return flows;
}


// Computes flow estimates using the DSF procedure - see page 373 of "Gravity Models of Spatial Interaction Behavior" book.
double *infer_flows_dsf_procedure(double *flows,
                                const int *orig_supply,
                                const int *dest_demand,
                                const double *cost_mat,
                                const size_t n,
                                const size_t m,
                                const double beta,
                                const size_t max_iters,
                                const bool show_params,
                                const bool show_flows) {

    /*
    orig_supply [array]: supply generated at each origin zone
    dest_demand [array]: demand generated at each destination zone
    cost_mat [array]: matrix of costs travelling from each origin to each destination
    n [int]: number of origin zones
    m [int]: number of destination zones
    beta [double]: parameter of model controlling effect of cost matrix on the flow update
    max_iters [int]: maximum iterations for which system of equations should be solved
    show_params [boolean]: flag for printing parameter matrices A and B
    show_flows [boolean]: flag for printing flow from each origin to each destination
    */


    size_t i,ii,j,jj;
    unsigned int t;
    double temp;


    for(i=0; i<n; i++) {

        for(j=0; j<m; j++) {
          // Initialise flows
          // See equation (5.61) page 373 from Gravity Models of Spatial Interaction Behavior book
          // Assume Q_j = P_i = 1 for all i,j
          flows[i*m + j] = exp(-beta*cost_mat[i*m + j]);
        }
    }

    // Solve iteratively  'max_iters' times
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
      t++;

    }
    return flows;
}

/* Computer C(\beta) for a given choice of beta */
double cost_of_beta(double *flows,
                    const int *orig_supply,
                    const int *dest_demand,
                    const double *cost_mat,
                    const size_t n,
                    const size_t m,
                    double beta,
                    const size_t max_iters) {

  size_t i,j;

  // Infer flows from dsf procedure
  double *dsf_flows = infer_flows_dsf_procedure(flows,
                                        orig_supply,
                                        dest_demand,
                                        cost_mat,
                                        n,
                                        m,
                                        beta,
                                        max_iters,
                                        false,
                                        false);

  // Compute C(\beta)
  double c_beta = 0;
  for(i=0; i<n; i++) {
      for(j=0; j<m; j++) {
        c_beta += cost_mat[i*m + j] * dsf_flows[i*m + j];
      }
  }
  return c_beta;

}


// Computes flow estimates using the DSF procedure - see page 386 of "Gravity Models of Spatial Interaction Behavior" book.
void infer_flows_newton_raphson(double *flows,
                                double *beta,
                                double *c_beta,
                                const int *orig_supply,
                                const int *dest_demand,
                                const double *cost_mat,
                                const double total_cost,
                                const size_t n,
                                const size_t m,
                                const size_t dsf_max_iters,
                                const size_t newton_raphson_max_iters) {
    size_t i,j;
    int r;
    double num,denum,temp;

    /*
    flows [array]: array of flows that will be inferred
    beta [array]: parameters of model controlling effect of cost matrix on the flow update by iteration
    orig_supply [array]: supply generated at each origin zone
    dest_demand [array]: demand generated at each destination zone
    cost_mat [array]: matrix of costs travelling from each origin to each destination
    n [int]: number of origin zones
    m [int]: number of destination zones
    max_iters [int]: maximum iterations for which system of equations should be solved
    */

    // Initialise C(\beta) array
    for(r=0; r<newton_raphson_max_iters; r++) {
     c_beta[r] = -1.0;
    }

    // printf("Initial flows");
    // printf("\n");
    // printf("Flow[3,3]: ");
    // printf("%f",flows[i*3 + 3]);
    // printf("\n");

    // Compute C(\beta^{(0)})
    c_beta[0] = cost_of_beta(flows,
                              orig_supply,
                              dest_demand,
                              cost_mat,
                              n,
                              m,
                              beta[0],
                              dsf_max_iters);

    // Compute \beta^{(1)} = \beta^{(0)} C(\beta^{(0)})/C
    beta[1] = beta[0] * c_beta[0] * (1. / total_cost);

    // Compute \beta^{(r)} , C(\beta^{(r)}) for r = 2,3,...
    for(r=2; r<newton_raphson_max_iters; r++) {
        printf("Iteration:  ");
        printf("%d",r);
        printf(" / ");
        printf("%zu",newton_raphson_max_iters);
        printf("\n");

        // Update C(\beta^{(r-1)}), C(\beta^{(r-2)}) if necessary
        if (c_beta[r-1] == -1.0) {
          c_beta[r-1] = cost_of_beta(flows,
                                    orig_supply,
                                    dest_demand,
                                    cost_mat,
                                    n,
                                    m,
                                    beta[r-1],
                                    dsf_max_iters);
        }
        if (c_beta[r-2] == -1.0) {
          c_beta[r-2] = cost_of_beta(flows,
                                    orig_supply,
                                    dest_demand,
                                    cost_mat,
                                    n,
                                    m,
                                    beta[r-2],
                                    dsf_max_iters);
        }
        // printf("After iteration r = ");
        // printf("%d",r);
        // printf("\n");
        // printf("Flow[3,3]: ");
        // printf("%f",flows[i*3 + 3]);
        // printf("\n");

      // Update \beta^{(r)}

      // Compute numerator
      num = (c_beta[r-1]-total_cost) * beta[r-2]  -  (c_beta[r-2]-total_cost) * beta[r-1];
      // Compute denumerator
      denum = (c_beta[r-1]-c_beta[r-2]);
      // Store beta to temporary value
      temp = num / denum;

      // Check if beta has overflowed before storing its value
      if ( isinf(temp) || ! isfinite(temp) ) {
          // Stop loop
          break;
      }

      // Store beta
      beta[r] = temp;

    }

    // Infer flows from dsf procedure
    flows = infer_flows_dsf_procedure(flows,
                                      orig_supply,
                                      dest_demand,
                                      cost_mat,
                                      n,
                                      m,
                                      beta[r],
                                      dsf_max_iters,
                                      false,
                                      false);

    // printf("Resulting flows");
    // printf("\n");
    // printf("Flow[3,3]: ");
    // printf("%f",flows[i*3 + 3]);
    // printf("\n");
    // printf("DSF_Flow[3,3]: ");
    // printf("%f",dsf_flows[i*3 + 3]);
    // printf("\n");

}
