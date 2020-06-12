#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double potential_stochastic(const double *xx, double *jacobian, const double *dest_demand, const double *cost_mat, const double *theta, const size_t n, const size_t m)
{
    const double alpha = theta[0];
    const double gamma = theta[3];
    const double kappa = theta[4];
    const double epsilon = theta[5];
    const double alpha_inv = 1./alpha;

    size_t i, j;
    double temp;
    double value = 0.;

    // Initialise gradient/ Jacobian
    for(j = 0; j < m; ++j)
        jacobian[j] = 0.;

    for(j = 0; j < m; ++j) {
        // temp = - \epsilon \sum_{j=1}^M (\kappa / \alpha) * \exp(\alpha x_j) - D_j x_j
        // Equation 2
        value += ( kappa*alpha_inv*exp(alpha * xx[j]) - dest_demand[j]*xx[j] );
        jacobian[j] = - gamma*epsilon * (kappa*exp(alpha * xx[j]) - dest_demand[j]);
    }
    value *= -gamma*epsilon;

    return value;
}



void hessian_stochastic(const double *xx, double *hessian, const double *dest_demand, const double *cost_mat, const double *theta, const size_t n, const size_t m)
{
    const double alpha = theta[0];
    const double kappa = theta[4];
    const double epsilon = theta[5];
    const double alpha_inv = 1./alpha;

    size_t i, j, k;
    double temp;

    // Compute Hessian
    for(j = 0; j < m; ++j)
        hessian[j] = -epsilon*kappa*alpha_inv*exp(alpha*xx[j]);

}
