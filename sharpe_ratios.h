#ifndef SHARPE_RATIO_H
#define SHARPE_RATIO_H

#include <dlib/matrix.h>
//#include <dlib/rand.h>
//#include <dlib/matrix/matrix_la.h>
//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <asserts.h>
#include <cassert>
#include <kernel.h>

using namespace std;
using namespace dlib;

class Sharpe {
	public:
		Sharpe () {   };//default constructor
		~Sharpe () {   };//default destructor

		/* Calculate the Sharpe ratio */
		static double sharpe_ratio(const matrix<double> &X /* a vector of data */);

		/* Calculate the maximum squared Sharpe ratio that is attainable from a set of factors */
		static double max_sq_sharpe_ratio(const matrix<double> &X /* a T by N matrix of factor returns */);

		/* Calculate the maximum squared Sharpe ratio for mimicking portfolios of factors [see Section III in Barillas, Kan, Robotti & Shanken (2020)] */
		static double max_sq_sharpe_ratio_mimicking(const matrix<double> &X, /* a T by N matrix of non-traded factor returns */
													const matrix<double> &Z /* a T by M matrix of traded factor and basis asset returns */);


		/* Calculate the standard error of the difference between two Sharpe ratios */
		template<double kernel(double) /* a HAC kernel */>
		static double se_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
								const matrix<double> &Y, /* excess returns on second asset */
								const double bw /* kernel bandwidth */);

		/* Calculate the bootstrap standard error of the difference between two Sharpe ratios */
		static double boot_se_diff_sharpes(const matrix<double> &X_bt, /* bootstrapped excess returns on first asset */
											const matrix<double> &Y_bt, /* bootstrapped excess returns on second asset */
											const int B /* block size */);

		/* Calculate the standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of factors
		[see Proposition 1 of Barillas, Kan, Robotti & Shanken (2020)] */
		template<double kernel(double) /* a HAC kernel */>
		static double se_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
											  const matrix<double> &Y, /* a T by N2 matrix of factors */
											  const double bw /* kernel bandwidth */);

		/* Calculate the bootstrap standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of factors
		[see Proposition 1 of Barillas, Kan, Robotti & Shanken (2020) and Politis & Romano (1992)] */
		static double boot_se_diff_max_sq_sharpes(const matrix<double> &X_bt, /* a T by N1 matrix of bootstrapped factors */
												  const matrix<double> &Y_bt, /* a T by N2 matrix of bootstrapped factors */
												  const int B /* block size */);

		/* Calculate the standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
		[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
		template<double kernel(double) /* a HAC kernel */>
		static double se_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
													  const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
													  const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
													  const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
													  const double bw /* kernel bandwidth */);

		/* Calculate the bootstrap standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
		[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
		static double boot_se_diff_max_sq_sharpes_mimicking(const matrix<double> &X_bt, /* a T by N1 matrix of bootstrapped non-traded factors */
															const matrix<double> &Y_bt, /* a T by N2 matrix of bootstrapped non-traded factors */
															const matrix<double> &Z_x_bt, /* a T by M1 matrix of bootstrapped traded factors and basis asset returns */
															const matrix<double> &Z_y_bt, /* a T by M2 matrix of bootstrapped traded factors and basis asset returns */
															const int B /* block size */);

		/* Calculate the means of columns in a matrix */
		static matrix<double> mean(const matrix<double> &X /* a T by N matrix */);

		/* Calculate the sample covariance matrix of columns in a matrix */
		static matrix<double> covariance(const matrix<double> &X /* a T by N matrix */);

		/* Calculate the sample covariance matrix of two random vectors */
		static matrix<double> covariance(const matrix<double> &X, /* a T by N matrix */
										 const matrix<double> &Y /* a T by M matrix */);

	private:
		static matrix<double,4,1> func_f(const double &a, const double &b, const double &c, const double &d);


};

matrix<double,4,1> Sharpe::func_f(const double &a, const double &b, const double &c, const double &d) {
	matrix<double,4,1> grad;

	grad(0) = c/pow(c - pow(a,2.), 1.5);
	grad(1) = -d/pow(d - pow(b,2.), 1.5);
	grad(2) = -0.5*a/pow(c - pow(a,2.), 1.5);
	grad(3) = 0.5*b/pow(d - pow(b,2.), 1.5);

	return grad;

}

/* Calculate the means of columns in a matrix */
matrix<double> Sharpe::mean(const matrix<double> &X /* a T by N matrix */) {
	int N = X.nc();
	matrix<double> Y(N,1);

	for (int i = 0; i < N; ++i)
		Y(i) = dlib::mean(colm(X,i));
	return Y;
}

/* Calculate the sample covariance matrix of columns in a matrix */
matrix<double> Sharpe::covariance(const matrix<double> &X /* a T by N matrix */) {
	int N = X.nc(), T = X.nr();
	matrix<double> m(N,1), cov(N,N);

	m = Sharpe::mean(X);
	cov = trans(X)*X/T - m*trans(m);
	return cov;
}

/* Calculate the sample covariance matrix of two random vectors */
matrix<double> Sharpe::covariance(const matrix<double> &X, /* a T by N matrix */
								  const matrix<double> &Y /* a T by M matrix */) {
	assert(X.nr() == Y.nr());

	int T = X.nr(), N = X.nc(), M = Y.nc();
	matrix<double> mu_x(N,1), mu_y(M,1), cov(N,M);

	mu_x = Sharpe::mean(X);
	mu_y = Sharpe::mean(Y);
	cov = trans(X)*Y/T - mu_x*trans(mu_y);
	return cov;

}

double Sharpe::sharpe_ratio(const matrix<double> &X /* a vector of data */) {
	return mean(X)/stddev(X);
}


/* Calculate the maximum squared Sharpe ratio that is attainable from a set of factors */
double Sharpe::max_sq_sharpe_ratio(const matrix<double> &X /* a T by N matrix of factor returns */) {
	int N = X.nc();
	matrix<double> mu(N,1), cov(N,N);
	mu = Sharpe::mean(X);
	cov = Sharpe::covariance(X);

	return trans(mu)*inv(cov)*mu;
}

/* Calculate the maximum squared Sharpe ratio for mimicking portfolios of factors [see Section III in Barillas, Kan, Robotti & Shanken (2020)] */
double Sharpe::max_sq_sharpe_ratio_mimicking(const matrix<double> &X, /* a T by N matrix of non-traded factor returns */
											 const matrix<double> &Z /* a T by M matrix of traded factor and basis asset returns */) {
	assert(X.nr() == Z.nr());
	int T = X.nr(), N = X.nc(), M = Z.nc();

	matrix<double> mu(M,1), var(M,M), cov(M,N), A(N,1), V(N,N);

	mu = Sharpe::mean(Z);
	var = Sharpe::covariance(Z);
	cov = Sharpe::covariance(Z,X);
	A = (trans(cov)*inv(var))*mu;
	V = (trans(cov)*inv(var))*cov;

	double sr = (trans(A)*inv(V))*A;
	return sr;
}

/* Calculate the standard error of the difference between two Sharpe ratios */
template<double kernel(double) /* a HAC kernel */>
double Sharpe::se_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
								const matrix<double> &Y, /* excess returns on second asset */
								const double bw /* kernel bandwidth */) {

	ASSERT_(X.nr() == Y.nr());
	int T = X.nr();

	double mu1, mu2, gamma1, gamma2;
	mu1 = mean(X);
	mu2 = mean(Y);

	matrix<double> X2(T,1), Y2(T,1);
	X2 = pointwise_multiply(X,X);
	Y2 = pointwise_multiply(Y,Y);
	gamma1 = mean(X2);
	gamma2 = mean(Y2);
//	cout << mu1 << " " << mu2 << " " << gamma1 << " " << gamma2 << endl;

	matrix<double,4,1> grads;
	grads = Sharpe::func_f(mu1, mu2, gamma1, gamma2);
//	cout << grads << endl;

	matrix<double> nus(4,T);
	set_rowm(nus,0) = trans(X - mu1);
	set_rowm(nus,1) = trans(Y - mu2);
	set_rowm(nus,2) = trans(X2 - gamma1);
	set_rowm(nus,3) = trans(Y2 - gamma2);
//	cout << nus << endl;

	matrix<double,4,4> cov, cov_j;
	cov = 0.;
	for (int j = 1-T; j < T; ++j) {
		cov_j = 0.;
		if (j >= 0) {
			for (int t = j; t < T; ++t) {
				cov_j = cov_j + (colm(nus,t)*trans(colm(nus,t-j))/T);
			}
		}
		else {
			for (int t = -j; t < T; ++t) {
				cov_j = cov_j + (colm(nus,t+j)*trans(colm(nus,t))/T);
			}
		}
		cov = cov + (T*kernel(j/bw)*cov_j/(T-4));
	}
//	cout << cov << endl;

	double var = (trans(grads)*cov*grads)/T;
	return pow(var, 0.5);
}


/* Calculate the bootstrap standard error of the difference between two Sharpe ratios */
double Sharpe::boot_se_diff_sharpes(const matrix<double> &X_bt, /* bootstrapped excess returns on first asset */
									const matrix<double> &Y_bt, /* bootstrapped excess returns on second asset */
									const int B /* block size */) {

	ASSERT_(X_bt.nr() == Y_bt.nr());
	int T = X_bt.nr();
	int K = std::nearbyint((double) T/B);

	double mu1, mu2, gamma1, gamma2;
	mu1 = mean(X_bt);
	mu2 = mean(Y_bt);

	matrix<double> X2(T,1), Y2(T,1);
	X2 = pointwise_multiply(X_bt,X_bt);
	Y2 = pointwise_multiply(Y_bt,Y_bt);
	gamma1 = mean(X2);
	gamma2 = mean(Y2);
//	cout << mu1 << " " << mu2 << " " << gamma1 << " " << gamma2 << endl;

	matrix<double,4,1> grads;
	grads = Sharpe::func_f(mu1, mu2, gamma1, gamma2);
//	cout << grads << endl;

	matrix<double> nus(4,T);
	set_rowm(nus,0) = trans(X_bt - mu1);
	set_rowm(nus,1) = trans(Y_bt - mu2);
	set_rowm(nus,2) = trans(X2 - gamma1);
	set_rowm(nus,3) = trans(Y2 - gamma2);
//	cout << nus << endl;

	matrix<double,4,4> cov;
	matrix<double,4,1> xi_k1;
	cov = 0.;
	for (int k1 = 0; k1 < K; ++k1) {
		xi_k1 = 0.; //reset the values
		for (int t = 0; t < B; ++t) {
			xi_k1 = xi_k1 + ( (1./pow(B,0.5)) * colm(nus, k1*B+t) );
		}
		cov = cov + (xi_k1*trans(xi_k1)/K);
//		cout << cov << endl;
	}
//	cout << cov << endl;

	double var = (trans(grads)*cov*grads)/T;
	return pow(var, 0.5);
}


/* Calculate the standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of factors
[see Proposition 1 of Barillas, Kan, Robotti & Shanken (2020)] */
template<double kernel(double) /* a HAC kernel */>
double Sharpe::se_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
									  const matrix<double> &Y, /* a T by N2 matrix of factors */
									  const double bw /* kernel bandwidth */) {

	ASSERT_(X.nr() == Y.nr());
	int T = X.nr(), N1 = X.nc(), N2 = Y.nc();
	matrix<double> mu1(N1,1), mu2(N2,1);
	mu1 = Sharpe::mean(X);
	mu2 = Sharpe::mean(Y);

	matrix<double> cov1(N1,N1), cov2(N2,N2);
	cov1 = Sharpe::covariance(X);
	cov2 = Sharpe::covariance(Y);

	double max_sq_sharpe1 = Sharpe::max_sq_sharpe_ratio(X);
	double max_sq_sharpe2 = Sharpe::max_sq_sharpe_ratio(Y);

	matrix<double> u1(T,1), u2(T,1), d(T,1);
	for (int t = 0; t < T; ++t) {
		u1(t) = trans(mu1)*inv(cov1)*(trans(rowm(X,t)) - mu1);
		u2(t) = trans(mu2)*inv(cov2)*(trans(rowm(Y,t)) - mu2);
		d(t) = 2*(u1(t) - u2(t)) - (pow(u1(t),2.) - pow(u2(t),2.)) + (max_sq_sharpe1 - max_sq_sharpe2);
	}

	double cov_j = 0., var = 0.;
	for (int j = 1-T; j < T; ++j) {
		cov_j = 0.; //reset autocovariances
		if (j >= 0) {
			for (int t = j; t < T; ++t) {
				cov_j = cov_j + d(t)*d(t-j)/T;
			}
		}
		else {
			for (int t = -j; t < T; ++t) {
				cov_j = cov_j + d(t+j)*d(t)/T;
			}
		}
		var = var + (T*kernel(j/bw)*cov_j/(T-4));
	}
	return pow(var/T, 0.5);
}

/* Calculate the bootstrap standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of factors
[see Proposition 1 of Barillas, Kan, Robotti & Shanken (2020)] */
double Sharpe::boot_se_diff_max_sq_sharpes(const matrix<double> &X_bt, /* a T by N1 matrix of bootstrapped factors */
											const matrix<double> &Y_bt, /* a T by N2 matrix of bootstrapped factors */
											const int B /* block size */) {

	ASSERT_(X_bt.nr() == Y_bt.nr());
	int T = X_bt.nr(), N1 = X_bt.nc(), N2 = Y_bt.nc();
	int K = std::nearbyint((double) T/B);

	matrix<double> mu1(N1,1), mu2(N2,1);
	mu1 = Sharpe::mean(X_bt);
	mu2 = Sharpe::mean(Y_bt);

	matrix<double> cov1(N1,N1), cov2(N2,N2);
	cov1 = Sharpe::covariance(X_bt);
	cov2 = Sharpe::covariance(Y_bt);

	double max_sq_sharpe1 = Sharpe::max_sq_sharpe_ratio(X_bt);
	double max_sq_sharpe2 = Sharpe::max_sq_sharpe_ratio(Y_bt);

	matrix<double> u1(T,1), u2(T,1), d(T,1);
	for (int t = 0; t < T; ++t) {
		u1(t) = trans(mu1)*inv(cov1)*(trans(rowm(X_bt,t)) - mu1);
		u2(t) = trans(mu2)*inv(cov2)*(trans(rowm(Y_bt,t)) - mu2);
		d(t) = 2*(u1(t) - u2(t)) - (pow(u1(t),2.) - pow(u2(t),2.)) + (max_sq_sharpe1 - max_sq_sharpe2);
	}


	double var = 0., xi_k1 = 0.;
	for (int k1 = 0; k1 < K; ++k1) {
		xi_k1 = 0.; //reset the values
		for (int t = 0; t < B; ++t) {
			xi_k1 += (1./pow(B,0.5)) * d(k1*B+t);
		}
		var += pow(xi_k1, 2.)/K;
//		cout << var << endl;
	}
//	cout << var << endl;

	return pow(var/T, 0.5);
}


/* Calculate the standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
template<double kernel(double) /* a HAC kernel */>
double Sharpe::se_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
											  const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
											  const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
											  const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
											  const double bw /* kernel bandwidth */) {

	assert(X.nr() == Y.nr() && Z_x.nr() == X.nr() && Z_y.nr() == Y.nr());

	int T = X.nr(), N1 = X.nc(), N2 = Y.nc(), M1 = Z_x.nc(), M2 = Z_y.nc();
	matrix<double> mu_z1(N1,1), var_z1(M1,M1), cov_XZ(N1,M1), A_x(N1,M1), V_x(N1,N1), M_x(N1,1), X_x(T,N1), mu_x(M1,1);
	matrix<double> mu_z2(N2,1), var_z2(M2,M2), cov_YZ(N2,M2), A_y(N2,M2), V_y(N2,N2), M_y(N2,1), Y_y(T,N2), mu_y(M2,1);

	mu_z1 = Sharpe::mean(Z_x);
	var_z1 = Sharpe::covariance(Z_x);
	cov_XZ = Sharpe::covariance(X,Z_x);
	A_x = cov_XZ*inv(var_z1);
	V_x = (A_x*var_z1)*trans(A_x);
	M_x = A_x*mu_z1;
	X_x = Z_x*trans(A_x);
	mu_x = Sharpe::mean(X);

	mu_z2 = Sharpe::mean(Z_y);
	var_z2 = Sharpe::covariance(Z_y);
	cov_YZ = Sharpe::covariance(Y,Z_y);
	A_y = cov_YZ*inv(var_z2);
	V_y = (A_y*var_z2)*trans(A_y);
	M_y = A_y*mu_z2;
	Y_y = Z_y*trans(A_y);
	mu_y = Sharpe::mean(Y);

	double sr1 = Sharpe::max_sq_sharpe_ratio_mimicking(X,Z_x), sr2 = Sharpe::max_sq_sharpe_ratio_mimicking(Y,Z_y);

	matrix<double> eta1(N1,1), eta2(N2,1), d(T,1);
	double u1, y1, v1, h1, u2, y2, v2, h2;
	for (int t = 0; t < T; ++t) {
		u1 = (trans(M_x)*inv(V_x)) * (trans(rowm(X_x,t)) - M_x);
		eta1 = (trans(rowm(X,t)) - mu_x) - (trans(rowm(X_x,t)) - M_x);
		y1 = (trans(M_x)*inv(V_x)) * eta1;
		v1 = (trans(mu_z1)*inv(var_z1)) * (trans(rowm(Z_x,t)) - mu_z1);
		h1 = 2*u1*(1. - y1) - pow(u1,2.) + 2*y1*v1 + sr1;

		u2 = (trans(M_y)*inv(V_y)) * (trans(rowm(Y_y,t)) - M_y);
		eta2 = (trans(rowm(Y,t)) - mu_y) - (trans(rowm(Y_y,t)) - M_y);
		y2 = (trans(M_y)*inv(V_y)) * eta2;
		v2 = (trans(mu_z2)*inv(var_z2)) * (trans(rowm(Z_y,t)) - mu_z2);
		h2 = 2*u2*(1. - y2) - pow(u2,2.) + 2*y2*v2 + sr2;
		d(t) = h1 - h2;
	}

	double cov_j = 0., var = 0.;
	for (int j = 1-T; j < T; ++j) {
		cov_j = 0.; //reset autocovariances
		if (j >= 0) {
			for (int t = j; t < T; ++t) {
				cov_j = cov_j + d(t)*d(t-j)/T;
			}
		}
		else {
			for (int t = -j; t < T; ++t) {
				cov_j = cov_j + d(t+j)*d(t)/T;
			}
		}
		var = var + (T*kernel(j/bw)*cov_j/(T-4));
	}
	return pow(var/T, 0.5);
}

/* Calculate the bootstrap standard error of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
double Sharpe::boot_se_diff_max_sq_sharpes_mimicking(const matrix<double> &X_bt, /* a T by N1 matrix of bootstrapped non-traded factors */
													  const matrix<double> &Y_bt, /* a T by N2 matrix of bootstrapped non-traded factors */
													  const matrix<double> &Z_x_bt, /* a T by M1 matrix of bootstrapped traded factors and basis asset returns */
													  const matrix<double> &Z_y_bt, /* a T by M2 matrix of bootstrapped traded factors and basis asset returns */
													  const int B /* block size */) {

	assert(X_bt.nr() == Y_bt.nr() && Z_x_bt.nr() == X_bt.nr() && Z_y_bt.nr() == Y_bt.nr());

	int T = X_bt.nr(), N1 = X_bt.nc(), N2 = Y_bt.nc(), M1 = Z_x_bt.nc(), M2 = Z_y_bt.nc();
	matrix<double> mu_z1(N1,1), var_z1(M1,M1), cov_XZ(N1,M1), A_x(N1,M1), V_x(N1,N1), M_x(N1,1), X_x(T,N1), mu_x(M1,1);
	matrix<double> mu_z2(N2,1), var_z2(M2,M2), cov_YZ(N2,M2), A_y(N2,M2), V_y(N2,N2), M_y(N2,1), Y_y(T,N2), mu_y(M2,1);

	mu_z1 = Sharpe::mean(Z_x_bt);
	var_z1 = Sharpe::covariance(Z_x_bt);
	cov_XZ = Sharpe::covariance(X_bt,Z_x_bt);
	A_x = cov_XZ*inv(var_z1);
	V_x = (A_x*var_z1)*trans(A_x);
	M_x = A_x*mu_z1;
	X_x = Z_x_bt*trans(A_x);
	mu_x = Sharpe::mean(X_bt);

	mu_z2 = Sharpe::mean(Z_y_bt);
	var_z2 = Sharpe::covariance(Z_y_bt);
	cov_YZ = Sharpe::covariance(Y_bt,Z_y_bt);
	A_y = cov_YZ*inv(var_z2);
	V_y = (A_y*var_z2)*trans(A_y);
	M_y = A_y*mu_z2;
	Y_y = Z_y_bt*trans(A_y);
	mu_y = Sharpe::mean(Y_bt);

	double sr1 = Sharpe::max_sq_sharpe_ratio_mimicking(X_bt,Z_x_bt), sr2 = Sharpe::max_sq_sharpe_ratio_mimicking(Y_bt,Z_y_bt);

	matrix<double> eta1(N1,1), eta2(N2,1), d(T,1);
	double u1, y1, v1, h1, u2, y2, v2, h2;
	for (int t = 0; t < T; ++t) {
		u1 = (trans(M_x)*inv(V_x)) * (trans(rowm(X_x,t)) - M_x);
		eta1 = (trans(rowm(X_bt,t)) - mu_x) - (trans(rowm(X_x,t)) - M_x);
		y1 = (trans(M_x)*inv(V_x)) * eta1;
		v1 = (trans(mu_z1)*inv(var_z1)) * (trans(rowm(Z_x_bt,t)) - mu_z1);
		h1 = 2*u1*(1. - y1) - pow(u1,2.) + 2*y1*v1 + sr1;

		u2 = (trans(M_y)*inv(V_y)) * (trans(rowm(Y_y,t)) - M_y);
		eta2 = (trans(rowm(Y_bt,t)) - mu_y) - (trans(rowm(Y_y,t)) - M_y);
		y2 = (trans(M_y)*inv(V_y)) * eta2;
		v2 = (trans(mu_z2)*inv(var_z2)) * (trans(rowm(Z_y_bt,t)) - mu_z2);
		h2 = 2*u2*(1. - y2) - pow(u2,2.) + 2*y2*v2 + sr2;
		d(t) = h1 - h2;
	}

	int K = std::nearbyint((double) T/B);
	double var = 0., xi_k1 = 0.;
	for (int k1 = 0; k1 < K; ++k1) {
		xi_k1 = 0.; //reset the values
		for (int t = 0; t < B; ++t) {
			xi_k1 += (1./pow(B,0.5)) * d(k1*B+t);
		}
		var += pow(xi_k1, 2.)/K;
//		cout << var << endl;
	}
//	cout << var << endl;

	return pow(var/T, 0.5);
}
































#endif




































