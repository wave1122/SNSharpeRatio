#ifndef DGP_H
#define DGP_H

#include <dlib/matrix.h>
//#include <dlib/rand.h>
//#include <dlib/matrix/matrix_la.h>
//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <asserts.h>
#include <kernel.h>

using namespace std;
using namespace dlib;

using SampleType = matrix<double>;
using Samples = std::vector<SampleType>;

class Dgp {
	public:
		Dgp () {   };//default constructor
		~Dgp () {   };//default destructor


		// Function to print any dlib matrix as CSV (comma-separated values)
		template <typename T>
		static void print_matrix_csv(const matrix<T>& mat, ostream& out = cout);

		/* Draw random samples from a bivariate normal distribution */
		static std::pair<matrix<double>, matrix<double>> gen_bgaussian(const int num_samples, /* number of random samples */
																		const int T, /* sample size */
																		const double nu, /* parameter of the error distribution */
																		const matrix<double> &mu, /* means */
																		const matrix<double> &A, /* ARCH coefficients */
																		const matrix<double> &B, /* AR coefficients */
																		const matrix<double> &Omega, /* unconditional covariance matrix */
																		const string err_dist, /* distribution of the errors */
																		unsigned long seed /* a seed to generate random numbers */);

		/* Draw random samples from a multivariate normal distribution */
		static Samples gen_mgaussian(const int num_samples, /* number of random samples */
									const int T, /* sample size */
									const double nu, /* parameter of the error distribution */
									const matrix<double> &mu, /* means */
									const matrix<double> &A, /* ARCH coefficients */
									const matrix<double> &B, /* AR coefficients */
									const matrix<double> &Omega, /* unconditional covariance matrix */
									const string err_dist, /* distribution of the errors */
									unsigned long seed /* a seed to generate random numbers */);

		/* Draw random samples from a bivariate Student's distribution */
		static std::pair<matrix<double>, matrix<double>> gen_bstudent(const int num_samples, /* number of random samples */
																		const int T, /* sample size */
																		const double nu, /* parameter of the error distribution */
																		const matrix<double> &mu, /* means */
																		const matrix<double> &A, /* ARCH coefficients */
																		const matrix<double> &B, /* AR coefficients */
																		const matrix<double> &Omega, /* unconditional covariance matrix */
																		const string err_dist, /* distribution of the errors */
																		unsigned long seed /* a seed to generate random numbers */);

		/* Draw random samples from a multivariate Student's t distribution */
		static Samples gen_mstudent(const int num_samples, /* number of random samples */
									const int T, /* sample size */
									const double nu, /* parameter of the error distribution */
									const matrix<double> &mu, /* means */
									const matrix<double> &A, /* ARCH coefficients */
									const matrix<double> &B, /* AR coefficients */
									const matrix<double> &Omega, /* unconditional covariance matrix */
									const string err_dist, /* distribution of the errors */
									unsigned long seed /* a seed to generate random numbers */);

		/* Draw random samples from a diagonal-vech bivariate GARCH(1,1) model */
		template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
		static std::pair<matrix<double>, matrix<double>> gen_dvech_garch(const int num_samples, /* number of random samples */
																			const int T, /* sample size */
																			const double nu, /* parameter of the error distribution */
																			const matrix<double> &mu, /* means */
																			const matrix<double> &A, /* ARCH coefficients */
																			const matrix<double> &B, /* AR coefficients */
																			const matrix<double> &C, /* intercepts */
																			const string err_dist = "Student", /* distribution of the errors */
																			unsigned long seed = 123456 /* a seed to generate random numbers */);

		/* Draw random samples from a diagonal-vech bivariate AR(1)-GARCH(1,1) model */
		template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
		static std::pair<matrix<double>, matrix<double>> gen_dvech_argarch(const int num_samples, /* number of random samples */
																			const int T, /* sample size */
																			const double nu, /* parameter of the error distribution */
																			const matrix<double> &mu, /* means */
																			const matrix<double> &A, /* ARCH coefficients */
																			const matrix<double> &B, /* AR coefficients */
																			const matrix<double> &C, /* intercepts */
																			const string err_dist = "Student", /* distribution of the errors */
																			unsigned long seed = 123456 /* a seed to generate random numbers */);

		/* Draw random samples from a diagonal-vech multivariate GARCH(1,1) model */
		template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
		static Samples gen_dvech_mgarch(const int num_samples, /* number of random samples */
										const int T, /* sample size */
										const double nu, /* parameter of the error distribution */
										const matrix<double> &mu, /* means */
										const matrix<double> &A, /* ARCH coefficients */
										const matrix<double> &B, /* AR coefficients */
										const matrix<double> &Omega, /* unconditional covariance matrix */
										const string err_dist, /* distribution of the errors */
										unsigned long seed = 123456 /* a seed to generate random numbers */);

		/* Draw random samples from a diagonal-vech multivariate AR(1)-GARCH(1,1) model */
		template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
		static Samples gen_dvech_margarch(const int num_samples, /* number of random samples */
											const int T, /* sample size */
											const double nu, /* parameter of the error distribution */
											const matrix<double> &mu, /* means */
											const matrix<double> &A, /* ARCH coefficients */
											const matrix<double> &B, /* AR coefficients */
											const matrix<double> &Omega, /* unconditional covariance matrix */
											const string err_dist, /* distribution of the errors */
											unsigned long seed /* a seed to generate random numbers */);

		/* Draw random samples from a diagonal-vech trivariate GARCH(1,1) while imposing the constraints: mu_2 = V_21*V_11^{-1}*mu1 and m_3 = V_31*V_11^{-1}*mu1.
		[See Remarks and Proposition 5 of Barillas, Kan, Robotti & Shanken (2020)] */
		template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
		static Samples gen_dvech_trigarch(const int num_samples, /* number of random samples */
											const int T, /* sample size */
											const double nu, /* parameter of the error distribution */
											const matrix<double> &mu1, /* mean of the overlapping factor */
											const matrix<double> &A, /* ARCH coefficients */
											const matrix<double> &B, /* AR coefficients */
											const matrix<double> &Omega, /* unconditional covariance matrix */
											const string err_dist, /* distribution of the errors */
											unsigned long seed = 123456 /* a seed to generate random numbers */);


		/* Generate random samples of non-traded factors from random samples of traded factors */
		template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random innovations */>
		static Samples gen_non_traded_factors(Samples X, /* random samples of traded factors and test assets */
												const double nu, /* parameter of the error distribution */
												const matrix<double> &mu, /* means */
												const matrix<double> &A, /* Regression coefficients */
												const matrix<double> &Omega, /* unconditional covariance matrix */
												const string err_dist, /* distribution of the errors */
												unsigned long seed = 123456 /* a seed to generate random numbers */);
};


// Function to print any dlib matrix as CSV (comma-separated values)
template <typename T>
void Dgp::print_matrix_csv(const matrix<T>& mat, ostream& out) {
    for (long r = 0; r < mat.nr(); ++r) {
        for (long c = 0; c < mat.nc(); ++c) {
            out << mat(r, c);
            if (c < mat.nc() - 1)
                out << ",";  // Comma between columns but not after last
        }
        out << "\n";       // New line after each row
    }
}

/* Draw random samples from a multivariate normal distribution */
Samples Dgp::gen_mgaussian(const int num_samples, /* number of random samples */
							const int T, /* sample size */
							const double nu, /* parameter of the error distribution */
							const matrix<double> &mu, /* means */
							const matrix<double> &A, /* ARCH coefficients */
							const matrix<double> &B, /* AR coefficients */
							const matrix<double> &Omega, /* unconditional covariance matrix */
							const string err_dist, /* distribution of the errors */
							unsigned long seed /* a seed to generate random numbers */) {

	(void) nu, A, B, err_dist; //unused arguments
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    ASSERT_(mu.nr() == Omega.nr());

    int N = Omega.nr();
    Samples X1(num_samples, matrix<double>(T,N));

    matrix<double> L = chol(Omega); // Cholesky decomposition
    matrix<double> Z(N,1);
    for (int i = 0; i < num_samples; ++i) {
    	for (int t = 0; t < T; ++t) {
			for (int j = 0; j < N; ++j) {
				Z(j) = gsl_ran_ugaussian(r);
			}
			set_rowm(X1[i], t) = trans(L*Z + mu);
    	}
    }

	gsl_rng_free(r); //free memory

    return X1;
}

std::pair<matrix<double>, matrix<double>> Dgp::gen_bgaussian(const int num_samples, /* number of random samples */
															const int T, /* sample size */
															const double nu, /* parameter of the error distribution */
															const matrix<double> &mu, /* means */
															const matrix<double> &A, /* ARCH coefficients */
															const matrix<double> &B, /* AR coefficients */
															const matrix<double> &Omega, /* unconditional covariance matrix */
															const string err_dist, /* distribution of the errors */
															unsigned long seed /* a seed to generate random numbers */) {
	(void) nu, A, B, err_dist; //unused arguments
	assert(Omega.nr() == 2);

	auto Z = Dgp::gen_mgaussian(num_samples, /* number of random samples */
								T, /* sample size */
								nu, /* parameter of the error distribution */
								mu, /* means */
								A, /* ARCH coefficients */
								B, /* AR coefficients */
								Omega, /* unconditional covariance matrix */
								err_dist, /* distribution of the errors */
								seed /* a seed to generate random numbers */);

	matrix<double> X(T, num_samples), Y(T, num_samples);
	for (int i = 0; i < num_samples; ++i) {
		set_colm(X, i) = colm(Z[i], 0);
		set_colm(Y, i) = colm(Z[i], 1);
	}

	return {X, Y};

}

/* Draw random samples from a multivariate Student's t distribution */
Samples Dgp::gen_mstudent(const int num_samples, /* number of random samples */
							const int T, /* sample size */
							const double nu, /* parameter of the error distribution */
							const matrix<double> &mu, /* means */
							const matrix<double> &A, /* ARCH coefficients */
							const matrix<double> &B, /* AR coefficients */
							const matrix<double> &Omega, /* unconditional covariance matrix */
							const string err_dist, /* distribution of the errors */
							unsigned long seed /* a seed to generate random numbers */) {

	(void) A, B, err_dist; //unused arguments
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    ASSERT_(mu.nr() == Omega.nr() && nu > 2.);

    int N = Omega.nr();
    Samples X1(num_samples, matrix<double>(T,N));

	matrix<double> L = chol(Omega); // Cholesky decomposition
    matrix<double> Z(N,1);
    double chi2 = 0.;
    for (int i = 0; i < num_samples; ++i) {
    	for (int t = 0; t < T; ++t) {
			for (int j = 0; j < N; ++j) {
				Z(j) = gsl_ran_ugaussian(r); //generate Gaussian random variables
			}
			chi2 = gsl_ran_chisq(r,nu); //generate a Chi^2 random variable
			set_rowm(X1[i], t) = trans(mu + L*Z/sqrt(chi2/(nu-2)));
    	}
    }

	gsl_rng_free(r); //free memory

    return X1;
}

std::pair<matrix<double>, matrix<double>> Dgp::gen_bstudent(const int num_samples, /* number of random samples */
															const int T, /* sample size */
															const double nu, /* parameter of the error distribution */
															const matrix<double> &mu, /* means */
															const matrix<double> &A, /* ARCH coefficients */
															const matrix<double> &B, /* AR coefficients */
															const matrix<double> &Omega, /* unconditional covariance matrix */
															const string err_dist, /* distribution of the errors */
															unsigned long seed /* a seed to generate random numbers */) {
	(void) A, B, err_dist; //unused arguments
	assert(Omega.nr() == 2);

	auto Z = Dgp::gen_mstudent(num_samples, /* number of random samples */
								T, /* sample size */
								nu, /* parameter of the error distribution */
								mu, /* means */
								A, /* ARCH coefficients */
								B, /* AR coefficients */
								Omega, /* unconditional covariance matrix */
								err_dist, /* distribution of the errors */
								seed /* a seed to generate random numbers */);

	matrix<double> X(T, num_samples), Y(T, num_samples);
	for (int i = 0; i < num_samples; ++i) {
		set_colm(X, i) = colm(Z[i], 0);
		set_colm(Y, i) = colm(Z[i], 1);
	}

	return {X, Y};

}

/* Draw random samples from a diagonal-vech bivariate GARCH(1,1) model */
template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
std::pair<matrix<double>, matrix<double>> Dgp::gen_dvech_garch(const int num_samples, /* number of random samples */
																const int T, /* sample size */
																const double nu, /* parameter of the error distribution */
																const matrix<double> &mu, /* means */
																const matrix<double> &A, /* ARCH coefficients */
																const matrix<double> &B, /* AR coefficients */
																const matrix<double> &C, /* intercepts */
																const string err_dist, /* distribution of the errors */
																unsigned long seed /* a seed to generate random numbers */){
	if (err_dist == "Student") {
		ASSERT_(nu > 2.);
	}

	int burn = 50; //burning the first 50 observations

	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    matrix<double> X(T+burn, num_samples), Y(T+burn, num_samples);
    matrix<double> H0(2, 2), H1(2, 2), L(2, 2), errors(2, 1), Z(2, 1);

    double X0 = gsl_ran_dist(r, nu), Y0 = gsl_ran_dist(r, nu); //initial values
    if (err_dist == "Student") {
    	X0 = X0/pow(nu/(nu-2),0.5);
		Y0 = Y0/pow(nu/(nu-2),0.5);
    }
    for (int i = 0; i < num_samples; ++i) {
    	/* Initialize the mean processes and their conditional (co)variance processes */
		X(0,i) = X0;
		Y(0,i) = Y0;
		H0(0,0) = 1; //fabs(X0);
		H0(1,1) = 1; //fabs(Y0);
		H0(0,1) = 0.; //cross correlation is zero
		H0(1,0) = 0.;
		/* Generate a random sample */
		for (int t = 1; t < T+burn; ++t) {
			//generate conditional covariance matrix
			H1(0,0) = C(0,0) + A(0,0)*pow(X(t-1,i), 2.) + B(0,0)*H0(0,0);
			H1(1,1) = C(1,1) + A(1,1)*pow(Y(t-1,i), 2.) + B(1,1)*H0(1,1);
			H1(0,1) = C(0,1) + A(0,1)*X(t-1,i)*Y(t-1,i) + B(0,1)*H0(0,1);
			H1(1,0) = C(1,0) + A(1,0)*X(t-1,i)*Y(t-1,i) + B(1,0)*H0(1,0);

			//perform Cholesky decomposition: H1 = L * trans(L)
			L = chol(H1);

			//generate error terms
			errors(0) = gsl_ran_dist(r, nu);
			errors(1) = gsl_ran_dist(r, nu);
			if (err_dist == "Student") {
//				cout << "generate Student's t errors..." << endl;
				errors = errors/pow(nu/(nu-2),0.5);
			}

			//generate data
			Z = L*errors;
			X(t,i) = Z(0);
			Y(t,i) = Z(1);

			//reset conditional covariances
			H0 = H1;
		}
    }

//	std::tuple<matrix<double>, matrix<double>> outputs;
//	std::get<0>(outputs) = rowm(X,range(burn,T+burn-1)) + mu(0);
//	std::get<1>(outputs) = rowm(Y,range(burn,T+burn-1)) + mu(1);
//	outputs.first = rowm(X,range(burn,T+burn-1)) + mu(0);
//	outputs.second = rowm(Y,range(burn,T+burn-1)) + mu(1);

    gsl_rng_free(r); //free memory

    return {rowm(X,range(burn,T+burn-1)) + mu(0), rowm(Y,range(burn,T+burn-1)) + mu(1)};
}


/* Draw random samples from a diagonal-vech bivariate AR(1)-GARCH(1,1) model */
template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
std::pair<matrix<double>, matrix<double>> Dgp::gen_dvech_argarch(const int num_samples, /* number of random samples */
																const int T, /* sample size */
																const double nu, /* parameter of the error distribution */
																const matrix<double> &mu, /* means */
																const matrix<double> &A, /* ARCH coefficients */
																const matrix<double> &B, /* AR coefficients */
																const matrix<double> &C, /* intercepts */
																const string err_dist, /* distribution of the errors */
																unsigned long seed /* a seed to generate random numbers */){
	if (err_dist == "Student") {
		ASSERT_(nu > 2.);
	}

	int burn = 50; //burning the first 50 observations

	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    matrix<double> X(T+burn, num_samples), Y(T+burn, num_samples);
    matrix<double> H0(2, 2), H1(2, 2), L(2, 2), errors(2, 1), Z(2, 1);

    double X0 = gsl_ran_dist(r, nu), Y0 = gsl_ran_dist(r, nu); //initial values
    double X1 = gsl_ran_dist(r, nu), Y1 = gsl_ran_dist(r, nu); //initial values
    if (err_dist == "Student") {
    	X0 = X0/pow(nu/(nu-2),0.5);
		Y0 = Y0/pow(nu/(nu-2),0.5);
		X1 = X0/pow(nu/(nu-2),0.5);
		Y1 = Y0/pow(nu/(nu-2),0.5);
    }

    for (int i = 0; i < num_samples; ++i) {
    	/* Initialize the mean processes and their conditional (co)variance processes */
		X(0,i) = X0;
		X(1,i) = X1;
		Y(0,i) = Y0;
		Y(1,i) = Y1;
		H0(0,0) = 1; //fabs(X0);
		H0(1,1) = 1; //fabs(Y0);
		H0(0,1) = 0.; //cross correlation is zero
		H0(1,0) = 0.;
		/* Generate a random sample */
		for (int t = 2; t < T+burn; ++t) {
			//generate conditional covariance matrix
			H1(0,0) = C(0,0) + A(0,0)*pow(X(t-1,i) - A(0,0)*X(t-2,i), 2.) + B(0,0)*H0(0,0);
			H1(1,1) = C(1,1) + A(1,1)*pow(Y(t-1,i) - A(1,1)*Y(t-2,i), 2.) + B(1,1)*H0(1,1);
			H1(0,1) = C(0,1) + A(0,1)*(X(t-1,i) - A(0,0)*X(t-2,i))*(Y(t-1,i) - A(1,1)*Y(t-2,i)) + B(0,1)*H0(0,1);
			H1(1,0) = C(1,0) + A(1,0)*(X(t-1,i) - A(0,0)*X(t-2,i))*(Y(t-1,i) - A(1,1)*Y(t-2,i)) + B(1,0)*H0(1,0);

			//perform Cholesky decomposition: H1 = L * trans(L)
			L = chol(H1);

			//generate error terms
			errors(0) = gsl_ran_dist(r, nu);
			errors(1) = gsl_ran_dist(r, nu);
			if (err_dist == "Student") {
//				cout << "generate Student's t errors..." << endl;
				errors = errors/pow(nu/(nu-2),0.5);
			}

			//generate data
			Z = L*errors;
			X(t,i) = A(0,0)*X(t-1,i) + Z(0);
			Y(t,i) = A(1,1)*Y(t-1,i) + Z(1);

			//reset conditional covariances
			H0 = H1;
		}
    }

//	std::tuple<matrix<double>, matrix<double>> outputs;
//	std::get<0>(outputs) = rowm(X,range(burn,T+burn-1)) + mu(0);
//	std::get<1>(outputs) = rowm(Y,range(burn,T+burn-1)) + mu(1);
//	outputs.first = rowm(X,range(burn,T+burn-1)) + mu(0);
//	outputs.second = rowm(Y,range(burn,T+burn-1)) + mu(1);

    gsl_rng_free(r); //free memory

    return {rowm(X,range(burn,T+burn-1)) + mu(0), rowm(Y,range(burn,T+burn-1)) + mu(1)};
}


/* Draw random samples from a diagonal-vech trivariate GARCH(1,1) while imposing the constraints: mu_2 = V_21*V_11^{-1}*mu1 and m_3 = V_31*V_11^{-1}*mu1.
[See Remarks and Proposition 5 of Barillas, Kan, Robotti & Shanken (2020)] */
template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
Samples Dgp::gen_dvech_trigarch(const int num_samples, /* number of random samples */
								const int T, /* sample size */
								const double nu, /* parameter of the error distribution */
								const matrix<double> &mu1, /* mean of the overlapping factor */
								const matrix<double> &A, /* ARCH coefficients */
								const matrix<double> &B, /* AR coefficients */
								const matrix<double> &Omega, /* unconditional covariance matrix */
								const string err_dist, /* distribution of the errors */
								unsigned long seed /* a seed to generate random numbers */) {
	ASSERT_(Omega.nr() == 3 && Omega.nr() == Omega.nc());

	matrix<double> mu(3,1);
	mu(0) = mu1(0);
	mu(1) = Omega(1,0)*pow(Omega(0,0),-1.)*mu(0);
	mu(2) = Omega(2,0)*pow(Omega(0,0),-1)*mu(0);
//	cout << mu << endl;

	auto X = Dgp::gen_dvech_mgarch<gsl_ran_dist>(num_samples, /* number of random samples */
												 T, /* sample size */
												 nu, /* parameter of the error distribution */
												 mu, /* means */
												 A, /* ARCH coefficients */
												 B, /* AR coefficients */
												 Omega, /* unconditional covariance matrix */
												 err_dist, /* distribution of the errors */
												 seed /* a seed to generate random numbers */);
	return X;
}


/* Draw random samples from a diagonal-vech multivariate GARCH(1,1) model */
template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
Samples Dgp::gen_dvech_mgarch(const int num_samples, /* number of random samples */
								const int T, /* sample size */
								const double nu, /* parameter of the error distribution */
								const matrix<double> &mu, /* means */
								const matrix<double> &A, /* ARCH coefficients */
								const matrix<double> &B, /* AR coefficients */
								const matrix<double> &Omega, /* unconditional covariance matrix */
								const string err_dist, /* distribution of the errors */
								unsigned long seed /* a seed to generate random numbers */){
	if (err_dist == "Student") {
		ASSERT_(nu > 2.);
	}

	int N = A.nr(); //number of time series
	matrix<double> C(N,N);
	C = pointwise_multiply((ones_matrix<double>(N,N) - A - B), Omega);

	int burn = 50; //burning the first 50 observations

	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

//    matrix<double> X(T+burn, num_samples), Y(T+burn, num_samples);
    Samples X(num_samples, matrix<double>(T+burn,N)), X1(num_samples, matrix<double>(T,N));

    matrix<double> H0(N,N), H1(N,N), L(N,N), errors(N,1), Z(N,1);

    matrix<double> X0(1,N); //initial values
    for (int i = 0; i < N; ++i)
		X0(i) = gsl_ran_dist(r, nu);

    if (err_dist == "Student") {
    	X0 = X0/pow(nu/(nu-2),0.5);
    }
    for (int i = 0; i < num_samples; ++i) {
    	/* Initialize the mean processes and their conditional (co)variance processes */
		set_rowm(X[i],0) = X0;
		H0 = Omega;
		/* Generate a random sample */
		for (int t = 1; t < T+burn; ++t) {
			//generate conditional covariance matrix
			H1 = C + pointwise_multiply(A, trans(rowm(X[i],t-1))*rowm(X[i],t-1)) + pointwise_multiply(B, H0);

			//perform Cholesky decomposition: H1 = L * trans(L)
			L = chol(H1);

			//generate error terms
			for (int j = 0; j < N; ++j) {
				errors(j) = gsl_ran_dist(r, nu);
			}
			if (err_dist == "Student") {
//				cout << "generate Student's t errors..." << endl;
				errors = errors/pow(nu/(nu-2),0.5);
			}

			//generate data
			Z = L*errors;
			set_rowm(X[i],t) = trans(Z);

			//reset conditional covariances
			H0 = H1;
		}

		X1[i] = rowm(X[i],range(burn,T+burn-1));
		for (int j = 0; j < N; ++j) {
			set_colm(X1[i], j) = colm(X1[i], j) + mu(j);
		}
    }

    gsl_rng_free(r); //free memory

    return X1;
}

/* Draw random samples from a diagonal-vech multivariate AR(1)-GARCH(1,1) model */
template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random variables */>
Samples Dgp::gen_dvech_margarch(const int num_samples, /* number of random samples */
								const int T, /* sample size */
								const double nu, /* parameter of the error distribution */
								const matrix<double> &mu, /* means */
								const matrix<double> &A, /* ARCH coefficients */
								const matrix<double> &B, /* AR coefficients */
								const matrix<double> &Omega, /* unconditional covariance matrix */
								const string err_dist, /* distribution of the errors */
								unsigned long seed /* a seed to generate random numbers */){
	if (err_dist == "Student") {
		ASSERT_(nu > 2.);
	}

	int N = A.nr(); //number of time series
	matrix<double> C(N,N), A1(N,N), diag_elems(N,1);
	C = pointwise_multiply((ones_matrix<double>(N,N) - A - B), Omega);

	diag_elems = 0.1;
	A1 = diagm(diag_elems);  //set the AR coefficients for the return processes


	int burn = 50; //burning the first 50 observations

	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

//    matrix<double> X(T+burn, num_samples), Y(T+burn, num_samples);
    Samples X(num_samples, matrix<double>(T+burn,N)), Y(num_samples, matrix<double>(T,N));

    matrix<double> H0(N,N), H1(N,N), E(N,1), L(N,N), errors(N,1), Z(N,1);

    matrix<double> X0(1,N), X1(1,N), Xt(N,1); //initial values
    for (int i = 0; i < N; ++i) {
		X0(i) = gsl_ran_dist(r, nu);
		X1(i) = gsl_ran_dist(r, nu);
    }

    if (err_dist == "Student") {
    	X0 = X0/pow(nu/(nu-2),0.5);
    	X1 = X1/pow(nu/(nu-2),0.5);
    }
    for (int i = 0; i < num_samples; ++i) {
    	/* Initialize the mean processes and their conditional (co)variance processes */
		set_rowm(X[i],0) = X0;
		set_rowm(X[i],1) = X1;
		H0 = Omega;
		/* Generate a random sample */
		for (int t = 2; t < T+burn; ++t) {
			//generate conditional covariance matrix
			E = trans(rowm(X[i],t-1) - rowm(X[i],t-2)*A1);
			H1 = C + pointwise_multiply(A, E*trans(E)) + pointwise_multiply(B, H0);

			//perform Cholesky decomposition: H1 = L * trans(L)
			L = chol(H1);

			//generate error terms
			for (int j = 0; j < N; ++j) {
				errors(j) = gsl_ran_dist(r, nu);
			}
			if (err_dist == "Student") {
//				cout << "generate Student's t errors..." << endl;
				errors = errors/pow(nu/(nu-2),0.5);
			}

			//generate data
			Z = L*errors;
			Xt = A1*trans(rowm(X[i],t-1)) + Z;
			set_rowm(X[i],t) = trans(Xt);

			//reset conditional covariances
			H0 = H1;
		}

		Y[i] = rowm(X[i],range(burn,T+burn-1));
		for (int j = 0; j < N; ++j) {
			set_colm(Y[i], j) = colm(Y[i], j) + mu(j);
		}
    }

    gsl_rng_free(r); //free memory

    return Y;
}

/* Generate random samples of non-traded factors from random samples of traded factors */
template<double gsl_ran_dist(const gsl_rng *, double) /* a GLS function to generate random innovations */>
Samples Dgp::gen_non_traded_factors(Samples X, /* random samples of traded factors and test assets */
									const double nu, /* parameter of the error distribution */
									const matrix<double> &mu, /* means */
									const matrix<double> &A, /* Regression coefficients */
									const matrix<double> &Omega, /* unconditional covariance matrix */
									const string err_dist, /* distribution of the errors */
									unsigned long seed /* a seed to generate random numbers */) {
	if (err_dist == "Student") {
		assert(nu > 2.);
	}

	int num_samples = X.size(); //number of random samples
	int T = X[0].nr(); //number of time periods
	int N = X[0].nc(); //number of traded factors
	int M = A.nr(); //number of non-traded factors

	assert( mu.nr() == Omega.nr() && N == A.nc() && M == Omega.nr() && M == mu.nr() );
//	cout << num_samples << endl;

	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    Samples X1(num_samples, matrix<double>(T,M));

    matrix<double> L = chol(Omega); // Cholesky decomposition
    matrix<double> Z(M,1), H(M,1);
    for (int i = 0; i < num_samples; ++i) {
		for (int t = 0; t < T; ++t) {
			for (int j = 0; j < M; ++j) {
				//generate error terms
				Z(j) = gsl_ran_dist(r, nu);
			}
			if (err_dist == "Student") {
//				cout << "generate Student's t errors..." << endl;
				Z = Z/pow(nu/(nu-2),0.5);
			}
			H = A*trans(rowm(X[i],t));
			set_rowm(X1[i], t) = trans(mu + H + L*Z);

    	}
    }

    gsl_rng_free(r); //free memory

    return X1;
}

#endif
