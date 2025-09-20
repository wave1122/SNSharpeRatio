#ifndef BOOTSTRAPS_H
#define BOOTSTRAPS_H

#include <dlib/matrix.h>
//#include <dlib/rand.h>
//#include <dlib/matrix/matrix_la.h>
#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <asserts.h>
#include <kernel.h>

using namespace std;
using namespace dlib;


class Bootstrap {
	public:
		Bootstrap () {   };//default constructor
		~Bootstrap () {   };//default destructor

		/* Perform the circular block bootstrap algorithm of Politis & Romano (1992) */
		static matrix<double> CBB(const matrix<double> &X, /* a T by N matrix */
								  const int B, /* block size */
								  unsigned long seed = 12345 /* a seed to generate random numbers */);


};

matrix<double> Bootstrap::CBB(const matrix<double> &X, /* a T by N matrix */
							  const int B, /* block size */
							  unsigned long seed /* a seed to generate random numbers */) {

	int T = X.nr(), N = X.nc();
	int K = std::nearbyint((double) T/B);

	/* Define the periodically extended time series */
	matrix<double> Y(T+B,N);
	for (int t = 0; t < T+B; ++t){
		if (t < T)
			set_rowm(Y,t) = rowm(X,t);
		else
			set_rowm(Y,t) = rowm(X,t % T);
	}

	/* Draw random blocks of a fixed size B >= 1 */
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    matrix<double> X_boot(K*B,N);
    for (int k1 = 0; k1 < K; ++k1) {
		int I = gsl_rng_uniform_int(r,T);
//		cout << I << endl;
		for (int j = 0; j < B; ++j) {
			set_rowm(X_boot,k1*B+j) = rowm(Y,I+j);
		}
    }

	gsl_rng_free(r); //freee memory

	return X_boot;
}
































#endif
