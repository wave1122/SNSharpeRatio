#ifndef KERNEL_H
#define KERNEL_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>

using namespace std;


double daniell_kernel (double x)
{
	if (x != 0) return std::sin(M_PI * x)/(M_PI*x);
	else return 1.;
}

double QS_kernel (double x)
{
	if (x != 0) return (25./(12.*pow(M_PI*x, 2.))) * (std::sin(6*M_PI*x/5.)/(6*M_PI*x/5.) - std::cos(6*M_PI*x/5.));
	else return 1.;
}

double parzen_kernel (double x)
{
	if (std::fabs(x) <= 0.5) return 1 - 6*pow(x, 2.) + 6*pow(std::fabs(x), 3.);
	else if ((std::fabs(x) > 0.5) && (std::fabs(x) <= 1.)) return 2.*pow(1-std::fabs(x), 3.);
	else return 0;
}

double bartlett_kernel (double x)
{
	if (std::fabs(x) <= 1)
	    return 1. - std::fabs(x);
	else
	    return 0.;
}

double triangle_kernel (double x) {
	return std::max (1 - x, 0.);
}

double epanechnikov_kernel (double x) {
	return ((double) 3/4) * std::max (1 - pow(x,2.), 0.);
}

double gaussian_kernel (double x) {
	return exp(-pow(x,2.));
}




#endif
