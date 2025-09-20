#ifndef SN_TEST_SHARPE_H
#define SN_TEST_SHARPE_H

#include <unistd.h>
#include <omp.h>
#include <dlib/matrix.h>
#include <dlib/random_forest.h>

#include <iostream>
#include <random>
#include <regex>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <kernel.h>
#include <dgp.h>
#include <bootstraps.h>
#include <sharpe_ratios.h>

using namespace std;

class SnTestSharpeR{
	public:
		SnTestSharpeR (){  }; //default constructor
		~SnTestSharpeR () { };//default destructor

	/* Calculate the self-normalized test statistic for the difference between two Sharpe ratios */
	static double sn_sharpe_stat(const matrix<double> &X, /* excess returns on first asset */
								 const matrix<double> &Y /* excess returns on second asset */);

	/* Calculate Volgushev and Shao's (2014) self-normalized test statistic for the difference between two Sharpe ratios */
	static double sn2_sharpe_stat(const matrix<double> &X, /* excess returns on first asset */
								  const matrix<double> &Y /* excess returns on second asset */);


	/* Calculate the self-normalized test statistic for the difference between two maximum squared Sharpe ratios */
	static double sn_max_sq_sharpe_stat(const matrix<double> &X, /* a T by N1 matrix of factors */
										const matrix<double> &Y /* a T by N2 matrix of factors */);

	/* Calculate Volgushev and Shao's (2014) self-normalized test statistic for the difference between two maximum squared Sharpe ratios */
	static double sn2_max_sq_sharpe_stat(const matrix<double> &X, /* a T by N1 matrix of factors */
										 const matrix<double> &Y /* a T by N2 matrix of factors */);

	/* Calculate the self-normalized test statistic for the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
	[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
	static double sn_max_sq_sharpe_stat_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
												  const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
												  const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
												  const matrix<double> &Z_y /* a T by M2 matrix of traded factors and basis asset returns */);

	/* Calculate Volgushev and Shao's (2014) self-normalized test statistic for the difference between two maximum squared Sharpe ratios attainable from
	two sets of non-traded factors [see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
	static double sn2_max_sq_sharpe_stat_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
												  const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
												  const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
												  const matrix<double> &Z_y /* a T by M2 matrix of traded factors and basis asset returns */);

	private:
		static double func_f(const double &a, const double &b, const double &c, const double &d);

};

double SnTestSharpeR::func_f(const double &a, const double &b, const double &c, const double &d) {
	return (a/pow(c-pow(a,2.), 0.5)) - (b/pow(d-pow(b,2.), 0.5));
}

/* Calculate the self-normalized test statistic for the difference between two Sharpe ratios */
double SnTestSharpeR::sn_sharpe_stat(const matrix<double> &X, /* excess returns on first asset */
									 const matrix<double> &Y /* excess returns on second asset */) {
	ASSERT_(X.nr() == Y.nr());
	int T = X.nr();

	matrix<double>  sr_diff(T,1);
	double mu1 = 0., mu2 = 0., gamma1 = 0., gamma2 = 0., var1 = 0., var2 = 0.;

	double epsilon = 1e-2;

	int T1 = 0;
	for (int t = T1; t < T; ++t) {
//		cout << "t = " << t << endl;
		mu1 = dlib::mean(rowm(X, range(0,t)));
		mu2 = dlib::mean(rowm(Y, range(0,t)));
		gamma1 = dlib::mean(rowm(pointwise_multiply(X,X), range(0,t)));
		gamma2 = dlib::mean(rowm(pointwise_multiply(Y,Y), range(0,t)));
		var1 = gamma1 - pow(mu1, 2.);
		var2 = gamma2 - pow(mu2, 2.);
		
		if (fabs(var1) <= 0.) {
			gamma1 = gamma1+ epsilon;
		}
		if(fabs(var2) <= 0.) {
			gamma2 = gamma2 + epsilon;
		}

		sr_diff(t) = SnTestSharpeR::func_f(mu1, mu2, gamma1, gamma2);

	}

	double var = 0.;
	for (int t = T1; t < T; ++t) {
		var += pow( ((double) t/T)*(sr_diff(t) - sr_diff(T-1)), 2. );
//		cout << "var = " << var << endl;
	}

	return sqrt(T)*sr_diff(T-1)/sqrt(var);
}


/* Calculate Volgushev and Shao's (2014) self-normalized test statistic for the difference between two Sharpe ratios */
double SnTestSharpeR::sn2_sharpe_stat(const matrix<double> &X, /* excess returns on first asset */
									 const matrix<double> &Y /* excess returns on second asset */) {
	ASSERT_(X.nr() == Y.nr());
	int T = X.nr();

	double mu1 = 0., mu2 = 0., gamma1 = 0., gamma2 = 0., var1 = 0., var2 = 0.;

	double epsilon = 1e-2;

	matrix<double> sr_diff(T,T);
	int T1 = 0;
	for (int t = 0; t < T; ++t) {
		for (int s = t+T1; s < T; ++s) {
	//		cout << "t = " << t << endl;
			mu1 = dlib::mean(rowm(X, range(t,s)));
			mu2 = dlib::mean(rowm(Y, range(t,s)));
			gamma1 = dlib::mean(rowm(pointwise_multiply(X,X), range(t,s)));
			gamma2 = dlib::mean(rowm(pointwise_multiply(Y,Y), range(t,s)));
			var1 = gamma1 - pow(mu1, 2.);
			var2 = gamma2 - pow(mu2, 2.);

			if (fabs(var1) <= 0.) {
				gamma1 = gamma1 + epsilon;
			}
			if( fabs(var2) <= 0.) {
				gamma2 = gamma2 + epsilon;
			}

			sr_diff(t,s) = SnTestSharpeR::func_f(mu1, mu2, gamma1, gamma2);
		}
	}

	double var = 0.;
	for (int t = 0; t < T; ++t) {
		for (int s = t+T1; s < T; ++s) {
			var += pow( (s-t)*(sr_diff(t,s) - sr_diff(0,T-1)), 2.) / pow(T,3.);
		}
	}

	return sqrt(T)*sr_diff(0,T-1)/sqrt(var);
}

/* Calculate the self-normalized test statistic for the difference between two maximum squared Sharpe ratios */
double SnTestSharpeR::sn_max_sq_sharpe_stat(const matrix<double> &X, /* a T by N1 matrix of factors */
											const matrix<double> &Y /* a T by N2 matrix of factors */) {
	ASSERT_(X.nr() == Y.nr());
	int T = X.nr(), N1 = X.nc(), N2 = Y.nc();

	double epsilon = 1e-2;

	matrix<double> mu1(N1,1), mu2(N2,1), V1(N1,N1), V2(N2,N2), sr_diff(T,1);
//	double tau = 0.02; // tried: 0.01, 0.05, 0.005, 0.1, 0.5, 0.002
//	int T1 = std::nearbyint(T*tau);
	int T1 = 0;
	for (int t = T1; t < T; ++t) {
		mu1 = Sharpe::mean(rowm(X, range(0,t)));
		mu2 = Sharpe::mean(rowm(Y, range(0,t)));
		V1 = Sharpe::covariance(rowm(X, range(0,t)));
		V2 = Sharpe::covariance(rowm(Y, range(0,t)));

		if (fabs(det(V1)) <= 0.) {
			V1 = V1 + epsilon*identity_matrix<double>(V1.nr());
//			throw std::invalid_argument("Covariance matrix is not invertible!");
		}
		if (fabs(det(V2)) <= 0.) {
			V2 = V2 + epsilon*identity_matrix<double>(V2.nr());
		}

		sr_diff(t) = (trans(mu1)*inv(V1)*mu1) - (trans(mu2)*inv(V2)*mu2);
//		if ( std::isnan(sr_diff(t)) )
//			sr_diff(t) = 0.;
//		cout << sr_diff(t) << endl;
	}

	double var = 0.;
	for (int t = T1; t < T; ++t) {
		var += pow( ((double) t/T)*(sr_diff(t) - sr_diff(T-1)), 2. );
//		cout << "var = " << var << endl;
	}

	return sqrt(T)*sr_diff(T-1)/sqrt(var);
}

/* Calculate Volgushev and Shao's (2014) self-normalized test statistic for the difference between two maximum squared Sharpe ratios */
double SnTestSharpeR::sn2_max_sq_sharpe_stat(const matrix<double> &X, /* a T by N1 matrix of factors */
											 const matrix<double> &Y /* a T by N2 matrix of factors */) {
	ASSERT_(X.nr() == Y.nr());
	int T = X.nr(), N1 = X.nc(), N2 = Y.nc();

	double epsilon = 1e-2;

	matrix<double> mu1(N1,1), mu2(N2,1), V1(N1,N1), V2(N2,N2), sr_diff(T,T);
//	double tau = 0.02;
//	int T1 = std::nearbyint(T*tau);
	int T1  = 0.;
	for (int t = 0; t < T; ++t) {
		for (int s = t+T1; s < T; ++s) {
			mu1 = Sharpe::mean(rowm(X, range(t,s)));
			mu2 = Sharpe::mean(rowm(Y, range(t,s)));
			V1 = Sharpe::covariance(rowm(X, range(t,s)));
			V2 = Sharpe::covariance(rowm(Y, range(t,s)));

			if (fabs(det(V1)) <= 0.) {
				V1 = V1 + epsilon*identity_matrix<double>(V1.nr());
//				throw std::invalid_argument("Covariance matrix is not invertible!");
			}
			if (fabs(det(V2)) <= 0.) {
				V2 = V2 + epsilon*identity_matrix<double>(V2.nr());
			}

			sr_diff(t,s) = (trans(mu1)*inv(V1)*mu1) - (trans(mu2)*inv(V2)*mu2);

//			if ( std::isnan(sr_diff(t,s)) )
//				sr_diff(t,s) = 0.;
//			cout << sr_diff(t,s) << endl;
		}
	}

	double var = 0.;
	for (int t = 0; t < T; ++t) {
		for (int s = t+T1; s < T; ++s) {
			var += pow( (s-t)*(sr_diff(t,s) - sr_diff(0,T-1)), 2.) / pow(T,3.);
		}
	}

	return sqrt(T)*sr_diff(0,T-1)/sqrt(var);
}


/* Calculate the self-normalized test statistic for the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
double SnTestSharpeR::sn_max_sq_sharpe_stat_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
													  const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
													  const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
													  const matrix<double> &Z_y /* a T by M2 matrix of traded factors and basis asset returns */) {
	assert(X.nr() == Z_x.nr() && Y.nr() == Z_y.nr() && X.nr() == Y.nr());
	int T = X.nr(), N1 = X.nc(), M1 = Z_x.nc(), N2 = Y.nc(), M2 = Z_y.nc();

	matrix<double> mu1(M1,1), var1(M1,M1), cov1(M1,N1), A1(N1,1), V1(N1,N1);
	matrix<double> mu2(M2,1), var2(M2,M2), cov2(M2,N2), A2(N2,1), V2(N2,N2), sr_diff(T,1);

	double epsilon = 1e-2;

//	double tau = 0.02; // tried: 0.01, 0.05, 0.005, 0.1, 0.5, 0.002
//	int T1 = std::nearbyint(T*tau);
	int T1 = 0;
	for (int t = T1; t < T; ++t) {
		var1 = Sharpe::covariance(rowm(Z_x, range(0,t)));
		var2 = Sharpe::covariance(rowm(Z_y, range(0,t)));

		if (fabs(det(var1)) <= 0.) {
			var1 = var1 + epsilon*identity_matrix<double>(var1.nr());
//			throw std::invalid_argument("Covariance matrix is not invertible!");
		}
		if (fabs(det(var2)) <= 0.) {
			var2 = var2 + epsilon*identity_matrix<double>(var2.nr());
		}

		cov1 = Sharpe::covariance(rowm(Z_x, range(0,t)), rowm(X, range(0,t)));
		cov2 = Sharpe::covariance(rowm(Z_y, range(0,t)), rowm(Y, range(0,t)));
		V1 = (trans(cov1)*inv(var1))*cov1;
		V2 = (trans(cov2)*inv(var2))*cov2;

		if (fabs(det(V1)) <= 0.) {
			V1 = V1 + epsilon*identity_matrix<double>(V1.nr());
		}
		if( fabs(det(V2)) <= 0.) {
			V2 = V2 + epsilon*identity_matrix<double>(V2.nr());
		}

		mu1 = Sharpe::mean(rowm(Z_x, range(0,t)));
		mu2 = Sharpe::mean(rowm(Z_y, range(0,t)));
		A1 = (trans(cov1)*inv(var1))*mu1;
		A2 = (trans(cov2)*inv(var2))*mu2;
		sr_diff(t) = (trans(A1)*inv(V1)*A1) - (trans(A2)*inv(V2)*A2);

//		if ( std::isnan(sr_diff(t)) )
//			sr_diff(t) = 0.;
//		cout << sr_diff(t) << endl;
	}

	double var = 0.;
	for (int t = T1; t < T; ++t) {
		var += pow( ((double) t/T)*(sr_diff(t) - sr_diff(T-1)), 2. );
//		cout << "var = " << var << endl;
	}

	return sqrt(T)*sr_diff(T-1)/sqrt(var);
}


/* Calculate Volgushev and Shao's (2014) self-normalized test statistic for the difference between two maximum squared Sharpe ratios attainable from
two sets of non-traded factors [see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
double SnTestSharpeR::sn2_max_sq_sharpe_stat_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
													  const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
													  const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
													  const matrix<double> &Z_y /* a T by M2 matrix of traded factors and basis asset returns */) {
	assert(X.nr() == Z_x.nr() && Y.nr() == Z_y.nr() && X.nr() == Y.nr());
	int T = X.nr(), N1 = X.nc(), M1 = Z_x.nc(), N2 = Y.nc(), M2 = Z_y.nc();

	double epsilon = 1e-2;

	matrix<double> mu1(M1,1), var1(M1,M1), cov1(M1,N1), A1(N1,1), V1(N1,N1);
	matrix<double> mu2(M2,1), var2(M2,M2), cov2(M2,N2), A2(N2,1), V2(N2,N2), sr_diff(T,T);

//	double tau = 0.02;
//	int T1 = std::nearbyint(T*tau);
	int T1  = 0.;
	for (int t = 0; t < T; ++t) {
		for (int s = t+T1; s < T; ++s) {
			var1 = Sharpe::covariance(rowm(Z_x, range(t,s)));
			var2 = Sharpe::covariance(rowm(Z_y, range(t,s)));

			if (fabs(det(var1)) <= 0.) {
				var1 = var1 + epsilon*identity_matrix<double>(var1.nr());
	//			throw std::invalid_argument("Covariance matrix is not invertible!");
			}
			if (fabs(det(var2)) <= 0.) {
				var2 = var2 + epsilon*identity_matrix<double>(var2.nr());
			}

			cov1 = Sharpe::covariance(rowm(Z_x, range(t,s)), rowm(X, range(t,s)));
			cov2 = Sharpe::covariance(rowm(Z_y, range(t,s)), rowm(Y, range(t,s)));
			V1 = (trans(cov1)*inv(var1))*cov1;
			V2 = (trans(cov2)*inv(var2))*cov2;

			if (fabs(det(V1)) <= 0.) {
				V1 = V1 + epsilon*identity_matrix<double>(V1.nr());
			}
			if(fabs(det(V2)) <= 0.) {
				V2 = V2 + epsilon*identity_matrix<double>(V2.nr());
			}

			mu1 = Sharpe::mean(rowm(Z_x, range(t,s)));
			mu2 = Sharpe::mean(rowm(Z_y, range(t,s)));
			A1 = (trans(cov1)*inv(var1))*mu1;
			A2 = (trans(cov2)*inv(var2))*mu2;
			sr_diff(t,s) = (trans(A1)*inv(V1)*A1) - (trans(A2)*inv(V2)*A2);

//			if ( std::isnan(sr_diff(t,s)) )
//				sr_diff(t,s) = 0.;
//			cout << sr_diff(t,s) << endl;
		}
	}

	double var = 0.;
	for (int t = 0; t < T; ++t) {
		for (int s = t+T1; s < T; ++s) {
			var += pow( (s-t)*(sr_diff(t,s) - sr_diff(0,T-1)), 2.) / pow(T,3.);
		}
	}

	return sqrt(T)*sr_diff(0,T-1)/sqrt(var);
}



















#endif
