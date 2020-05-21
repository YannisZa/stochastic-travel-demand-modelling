#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double logsumexp(const double *xx, const size_t nn)
{
    size_t i;
    double max = xx[0];

    for(i = 1; i < nn; ++i)
        if(xx[i] > max)
            max = xx[i];

    double sum = 0;
    for(i = 0; i < nn; ++i)
        sum += exp(xx[i] - max);

    return max + log(sum);
}


double potential_stochastic(const double *xx, double *grad, const double *orig, const double *cost_mat, const double *theta, const size_t nn, const size_t mm, double *wksp)
{
    const double alpha = theta[0];
    const double beta = theta[1];
    const double delta = theta[2];
    const double gamma = theta[3];
    const double kappa = theta[4];
    const double epsilon = theta[5];
    const double alpha_inv = 1./alpha;

    size_t i, j;
    double temp;

    double value = 0.;
    for(j = 0; j < mm; ++j)
        grad[j] = 0.;

    for(i=0; i < nn; ++i)
    {
        for(j = 0; j < mm; ++j)
            /* wksp[j] = \alpha x_j - \beta c_{ij} */
            wksp[j] = alpha*xx[j] - beta*cost_mat[i*mm + j];
        /* temp = \log (\sum_{j=1}^{M} exp(\alpha x_j - \beta c_{ij}) */
        temp = logsumexp(wksp, mm);
        /* Equation 2.7 */
        value += -epsilon*alpha_inv*orig[i]*temp;

        for(j = 0; j < mm; ++j)
            grad[j] += -epsilon*orig[i]*exp(wksp[j] - temp);
    }

    for(j = 0; j < mm; ++j)
    {
        /* Equation 2.22 */
        /* temp = \kappa \exp(x_j) */
        temp = kappa*exp(xx[j]);
        /* value = \kappa \exp(x_j) - \delta x_j*/
        value += temp - delta*xx[j];
        grad[j] += temp - delta;
    }


    value *= gamma;
    for(int j = 0; j < mm; ++j)
        grad[j] *= gamma;

    return value;
}


double potential_deterministic(const double *xx, double *grad, const double *orig, const double *cost_mat, const double *theta, const size_t nn, const size_t mm, double *wksp)
{
    const double alpha = theta[0];
    const double beta = theta[1];
    const double delta = theta[2];
    const double kappa = theta[4];
    const double epsilon = theta[5];
    const double alpha_inv = 1./alpha;

    size_t i, j;
    double temp;

    double value = 0.;
    for(j = 0; j < mm; ++j)
        grad[j] = 0.;

    for(i=0; i < nn; ++i)
    {
        for(j = 0; j < mm; ++j)
            wksp[j] = alpha*xx[j] - beta*cost_mat[i*mm + j];
        temp = logsumexp(wksp, mm);
        value += -epsilon*alpha_inv*orig[i]*temp;

        for(j = 0; j < mm; ++j)
            grad[j] += -epsilon*orig[i]*exp(wksp[j] - temp);
    }

    for(j = 0; j < mm; ++j)
    {
        temp = kappa*exp(xx[j]);
        value += temp - delta*xx[j];
        grad[j] += temp - delta;
    }

    return value;
}




void hessian_stochastic(const double *xx, double *hess, const double *orig, const double *cost_mat, const double *theta, const size_t nn, const size_t mm, double *wksp)
{
    const double alpha = theta[0];
    const double beta = theta[1];
    const double gamma = theta[3];
    const double kappa = theta[4];
    const double epsilon = theta[5];

    size_t i, j, k;
    double temp;

    for(j = 0; j < mm; ++j)
        for(k = 0; k < mm; ++k)
            hess[j*mm + k] = 0.;

    for(i=0; i < nn; ++i)
    {
        for(j = 0; j < mm; ++j)
            wksp[j] = alpha*xx[j] - beta*cost_mat[i*mm + j];
        temp = logsumexp(wksp, mm);

        for(j = 0; j < mm; ++j)
            wksp[j] = exp(wksp[j] - temp);


        for(j = 0; j < mm; ++j)
            for(k = j+1; k < mm; ++k)
            {
                temp = epsilon*alpha*orig[i]*wksp[j]*wksp[k];
                hess[j*mm + k] += temp;
                hess[k*mm + j] += temp;
            }

        for(j = 0; j < mm; ++j)
            hess[j*mm + j] += epsilon*alpha*orig[i]*wksp[j]*(wksp[j] - 1.);
    }

    for(j = 0; j < mm; ++j)
        hess[j*mm + j] += kappa*exp(xx[j]);

    for(j = 0; j < mm; ++j)
        for(k = 0; k < mm; ++k)
            hess[j*mm + k] *= gamma;

}


void hessian_deterministic(const double *xx, double *hess, const double *orig, const double *cost_mat, const double *theta, const size_t nn, const size_t mm, double *wksp)
{
    const double alpha = theta[0];
    const double beta = theta[1];
    const double kappa = theta[4];
    const double epsilon = theta[5];

    size_t i, j, k;
    double temp;

    for(j = 0; j < mm; ++j)
        for(k = 0; k < mm; ++k)
            hess[j*mm + k] = 0.;

    for(i=0; i < nn; ++i)
    {
        for(j = 0; j < mm; ++j)
            wksp[j] = alpha*xx[j] - beta*cost_mat[i*mm + j];
        temp = logsumexp(wksp, mm);

        for(j = 0; j < mm; ++j)
            wksp[j] = exp(wksp[j] - temp);


        for(j = 0; j < mm; ++j)
            for(k = j+1; k < mm; ++k)
            {
                temp = epsilon*alpha*orig[i]*wksp[j]*wksp[k];
                hess[j*mm + k] += temp;
                hess[k*mm + j] += temp;
            }

        for(j = 0; j < mm; ++j)
            hess[j*mm + j] += epsilon*alpha*orig[i]*wksp[j]*(wksp[j] - 1.);
    }

    for(j = 0; j < mm; ++j)
        hess[j*mm + j] += kappa*exp(xx[j]);

}
