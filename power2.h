#ifndef POWER_H
#define POWER_H

#include <unistd.h>
#include <omp.h>
#include <dlib/matrix.h>
#include <dlib/random_forest.h>
#include <dlib/statistics/running_gradient.h>

#include <iostream>
#include <random>
#include <regex>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <kernel.h>
#include <dgp.h>
#include <test_sharpe_ratios.h>
#include <sn_test_sharpe_ratios4.h>

using namespace std;
using namespace dlib;


class Power{
	public:
		Power (){  }; //default constructor
		~Power () { };//default destructor

		/* Calculate empirical rejection frequencies of Student's t-test, the bootstrap Student's t-test, and the self-normalized t-test
		for the null hypothesis of equality between two Sharpe ratios */
		template</* a HAC kernel */
		double kernel(double),
		/* a data generating process */
		std::pair<matrix<double>, matrix<double>> gen_data(const int, /* number of random samples */
															const int, /* sample size */
															const double, /* parameter of the error distribution */
															const matrix<double> &, /* means */
															const matrix<double> &, /* ARCH coefficients */
															const matrix<double> &, /* AR coefficients */
															const matrix<double> &, /* intercepts */
															const string,          /* distribution of the errors */
															unsigned long          /* a seed to generate random numbers */)>
		static std::tuple<matrix<double>, matrix<double>, matrix<double>,
							matrix<double>, matrix<double>, matrix<double>,
							double, double, double,
							double, double, double> power_f(const int num_samples, /* number of random samples */
															const int T, /* sample size */
															const matrix<double> &bws, /* kernel bandwidths */
															const matrix<int> &Ls, /* block sizes */
															const int num_boots, /* number of bootstrap repetitions */
															const double nu, /* parameter of the error distribution */
															const matrix<double> &mu, /* means */
															const matrix<double> &A, /* ARCH coefficients */
															const matrix<double> &B, /* AR coefficients */
															const matrix<double> &C, /* intercepts */
															const string err_dist, /* distribution of the errors */
															ofstream &pwr_out, /* output stream */
															unsigned long seed = 12345678 /* a seed to generate random numbers */);

		/* Calculate empirical critical values of Student's t-test and the self-normalized t-test for the null hypothesis of equality between two Sharpe ratios */
		template</* a HAC kernel */
				double kernel(double),
				/* a data generating process */
				std::pair<matrix<double>, matrix<double>> gen_data(const int, /* number of random samples */
																	const int, /* sample size */
																	const double, /* parameter of the error distribution */
																	const matrix<double> &, /* means */
																	const matrix<double> &, /* ARCH coefficients */
																	const matrix<double> &, /* AR coefficients */
																	const matrix<double> &, /* intercepts */
																	const string,          /* distribution of the errors */
																	unsigned long          /* a seed to generate random numbers */)>
		static std::tuple<matrix<double>, matrix<double>, matrix<double>,
							double, double, double,
							double, double, double> critical_vals(const int num_samples, /* number of random samples */
																	const int T, /* sample size */
																	const matrix<double> &bws, /* kernel bandwidths */
																	const double nu, /* parameter of the error distribution */
																	const matrix<double> &mu, /* means */
																	const matrix<double> &A, /* ARCH coefficients */
																	const matrix<double> &B, /* AR coefficients */
																	const matrix<double> &C, /* intercepts */
																	const string err_dist, /* distribution of the errors */
																	ofstream &pwr_out, /* output stream */
																	unsigned long seed /* a seed to generate random numbers */);

		/* Calculate empirical rejection rates using the empirical critical values of Student's t-test and the self-normalized t-tests
		for the null hypothesis of equality between two Sharpe ratios */
		template</* a HAC kernel */
				double kernel(double),
				/* a data generating process */
				std::pair<matrix<double>, matrix<double>> gen_data(const int, /* number of random samples */
																	const int, /* sample size */
																	const double, /* parameter of the error distribution */
																	const matrix<double> &, /* means */
																	const matrix<double> &, /* ARCH coefficients */
																	const matrix<double> &, /* AR coefficients */
																	const matrix<double> &, /* intercepts */
																	const string,          /* distribution of the errors */
																	unsigned long          /* a seed to generate random numbers */)>
		static std::tuple<matrix<double>, matrix<double>, matrix<double>,
					double, double, double,
					double, double, double> epower_f(const int num_samples, /* number of random samples */
													const int T, /* sample size */
													const matrix<double> &bws, /* kernel bandwidths */
													const double nu, /* parameter of the error distribution */
													const matrix<double> &t_cv1, /* 1%-level critical values of Student's t-test */
													const matrix<double> &t_cv5, /* 5%-level critical values of Student's t-test */
													const matrix<double> &t_cv10, /* 10%-level critical values of Student's t-test */
													const double sn_cv1, /* 1%-level critical value of the self-normalized test */
													const double sn_cv5, /* 5%-level critical value of the self-normalized test */
													const double sn_cv10, /* 10%-level critical value of the self-normalized test */
													const double sn2_cv1, /* 1%-level critical value of Volgushev and Shao's (2014) self-normalized test */
													const double sn2_cv5, /* 5%-level critical value of Volgushev and Shao's (2014) self-normalized test */
													const double sn2_cv10, /* 10%-level critical value of Volgushev and Shao's (2014) self-normalized test */
													const matrix<double> &mu, /* means */
													const matrix<double> &A, /* ARCH coefficients */
													const matrix<double> &B, /* AR coefficients */
													const matrix<double> &C, /* intercepts */
													const string err_dist, /* distribution of the errors */
													ofstream &pwr_out, /* output stream */
													unsigned long seed = 123456 /* a seed to generate random numbers */);

		/* Calculate empirical rejection frequencies of Student's t-test, the bootstrap t-test, and the self-normalized t-tests for the null hypothesis of
		equality between two maximum squared Sharpe ratios that are attainable from two overlapping sets of factors */
		template</* a HAC kernel */
				double kernel(double),
				/* a trivariate data generating process */
				Samples gen_data(const int, /* number of random samples */
								const int, /* sample size */
								const double, /* parameter of the error distribution */
								const matrix<double> &, /* means */
								const matrix<double> &, /* ARCH coefficients */
								const matrix<double> &, /* AR coefficients */
								const matrix<double> &, /* unconditional covariance matrix */
								const string,          /* distribution of the errors */
								unsigned long          /* a seed to generate random numbers */)>
		static std::tuple<matrix<double>, matrix<double>, matrix<double>,
							matrix<double>, matrix<double>, matrix<double>,
							double, double, double,
							double, double, double> power_fovl(const int num_samples, /* number of random samples */
																const int T, /* sample size */
																const matrix<double> &bws, /* kernel bandwidths */
																const matrix<int> &Ls, /* block sizes */
																const int num_boots, /* number of bootstrap repetitions */
																const double nu, /* parameter of the error distribution */
																const matrix<double> &mu, /* means */
																const matrix<double> &A, /* ARCH coefficients */
																const matrix<double> &B, /* AR coefficients */
																const matrix<double> &Omega, /* unconditional covariance matrix */
																const string err_dist, /* distribution of the errors */
																const bool fovl, /* True if there are overlapping factors */
																ofstream &pwr_out, /* output stream */
																unsigned long seed = 123456 /* a seed to generate random numbers */);

		/* Calculate empirical critical values of Student's t-test and the self-normalized t-tests for the null hypothesis of equality between
		two maximum squared Sharpe ratios that are attainable from two overlapping sets of factors */
		template</* a HAC kernel */
				double kernel(double),
				/* a trivariate data generating process */
				Samples gen_data(const int, /* number of random samples */
								const int, /* sample size */
								const double, /* parameter of the error distribution */
								const matrix<double> &, /* means */
								const matrix<double> &, /* ARCH coefficients */
								const matrix<double> &, /* AR coefficients */
								const matrix<double> &, /* unconditional covariance matrix */
								const string,          /* distribution of the errors */
								unsigned long          /* a seed to generate random numbers */)>
		static std::tuple<matrix<double>, matrix<double>, matrix<double>,
							double, double, double,
							double, double, double> critical_vals_fovl(const int num_samples, /* number of random samples */
																	const int T, /* sample size */
																	const matrix<double> &bws, /* kernel bandwidths */
																	const double nu, /* parameter of the error distribution */
																	const matrix<double> &mu, /* means */
																	const matrix<double> &A, /* ARCH coefficients */
																	const matrix<double> &B, /* AR coefficients */
																	const matrix<double> &Omega, /* unconditional covariance matrix */
																	const string err_dist, /* distribution of the errors */
																	const bool fovl, /* True if there are overlapping factors */
																	ofstream &pwr_out, /* output stream */
																	unsigned long seed = 123456 /* a seed to generate random numbers */);

		/* Calculate empirical rejection rates using the empirical critical values of Student's t-test and the self-normalized t-tests
		for the null hypothesis of equality between two maximum squared Sharpe ratios that are attainable from two overlapping sets of factors */
		template</* a HAC kernel */
				double kernel(double),
				/* a data generating process */
				Samples gen_data(const int, /* number of random samples */
								const int, /* sample size */
								const double, /* parameter of the error distribution */
								const matrix<double> &, /* means */
								const matrix<double> &, /* ARCH coefficients */
								const matrix<double> &, /* AR coefficients */
								const matrix<double> &, /* unconditional covariance matrix */
								const string,          /* distribution of the errors */
								unsigned long          /* a seed to generate random numbers */)>
		static std::tuple<matrix<double>, matrix<double>, matrix<double>,
							double, double, double,
							double, double, double> epower_fovl(const int num_samples, /* number of random samples */
																const int T, /* sample size */
																const matrix<double> &bws, /* kernel bandwidths */
																const double nu, /* parameter of the error distribution */
																const matrix<double> &t_cv1, /* 1%-level critical values of Student's t-test */
																const matrix<double> &t_cv5, /* 5%-level critical values of Student's t-test */
																const matrix<double> &t_cv10, /* 10%-level critical values of Student's t-test */
																const double sn_cv1, /* 1%-level critical value of the self-normalized test */
																const double sn_cv5, /* 5%-level critical value of the self-normalized test */
																const double sn_cv10, /* 10%-level critical value of the self-normalized test */
																const double sn2_cv1, /* 1%-level critical value of Volgushev and Shao's (2014) self-normalized test */
																const double sn2_cv5, /* 5%-level critical value of Volgushev and Shao's (2014) self-normalized test */
																const double sn2_cv10, /* 10%-level critical value of Volgushev and Shao's (2014) self-normalized test */
																const matrix<double> &mu, /* means */
																const matrix<double> &A, /* ARCH coefficients */
																const matrix<double> &B, /* AR coefficients */
																const matrix<double> &Omega, /* unconditional covariance matrix */
																const string err_dist, /* distribution of the errors */
																const bool fovl, /* True if there are overlapping factors */
																ofstream &pwr_out, /* output stream */
																unsigned long seed = 123456 /* a seed to generate random numbers */);

		/* Calculate empirical rejection frequencies of Student's t-test, the bootstrap t-test, and the self-normalized t-tests for the null hypothesis of
		equality between two maximum squared Sharpe ratios that are attainable from two sets of non-traded factors
		[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
		template</* a HAC kernel */
				double kernel(double),
				/* a trivariate data generating process */
				Samples gen_data(const int, /* number of random samples */
								const int, /* sample size */
								const double, /* parameter of the error distribution */
								const matrix<double> &, /* means */
								const matrix<double> &, /* ARCH coefficients */
								const matrix<double> &, /* AR coefficients */
								const matrix<double> &, /* unconditional covariance matrix */
								const string,          /* distribution of the errors */
								unsigned long          /* a seed to generate random numbers */)>
		static std::tuple<matrix<double>, matrix<double>, matrix<double>,
							matrix<double>, matrix<double>, matrix<double>,
							double, double, double,
							double, double, double> power_fovl_mimicking(const int num_samples, /* number of random samples */
																		const int T, /* sample size */
																		const matrix<double> &bws, /* kernel bandwidths */
																		const matrix<int> &Ls, /* block sizes */
																		const int num_boots, /* number of bootstrap repetitions */
																		const double nu, /* parameter of the error distribution */
																		const matrix<double> &mu1, /* means of traded factors and basis asset returns */
																		const matrix<double> &mu2, /* means of non-traded factors */
																		const matrix<double> &A, /* ARCH coefficients */
																		const matrix<double> &B, /* AR coefficients */
																		const matrix<double> &C, /* Regression coefficients used to generate non-traded factors */
																		const matrix<double> &Omega1, /* unconditional covariance matrx of the innovations used to generate traded factors */
																		const matrix<double> &Omega2, /* covariance matrix of the innovations used to generate non-traded factors */
																		const string err_dist, /* distribution of the errors */
																		const bool fovl, /* True if there are overlapping factors */
																		ofstream &pwr_out, /* output stream */
																		unsigned long seed = 123456 /* a seed to generate random numbers */);

};

template</* a HAC kernel */
		double kernel(double),
		/* a data generating process */
		std::pair<matrix<double>, matrix<double>> gen_data(const int, /* number of random samples */
															const int, /* sample size */
															const double, /* parameter of the error distribution */
															const matrix<double> &, /* means */
															const matrix<double> &, /* ARCH coefficients */
															const matrix<double> &, /* AR coefficients */
															const matrix<double> &, /* intercepts */
															const string,          /* distribution of the errors */
															unsigned long          /* a seed to generate random numbers */)>
std::tuple<matrix<double>, matrix<double>, matrix<double>,
			matrix<double>, matrix<double>, matrix<double>,
			double, double, double,
			double, double, double> Power::power_f( const int num_samples, /* number of random samples */
													const int T, /* sample size */
													const matrix<double> &bws, /* kernel bandwidths */
													const matrix<int> &Ls, /* block sizes */
													const int num_boots, /* number of bootstrap repetitions */
													const double nu, /* parameter of the error distribution */
													const matrix<double> &mu, /* means */
													const matrix<double> &A, /* ARCH coefficients */
													const matrix<double> &B, /* AR coefficients */
													const matrix<double> &C, /* intercepts */
													const string err_dist, /* distribution of the errors */
													ofstream &pwr_out, /* output stream */
													unsigned long seed /* a seed to generate random numbers */) {

	/* Generate many random samples */
    auto [X, Y] = gen_data(num_samples, /* number of random samples */
							T, /* sample size */
							nu, /* parameter of the error distribution */
							mu, /* means */
							A, /* ARCH coefficients */
							B, /* AR coefficients */
							C, /* intercepts */
							err_dist, /* distribution of the errors */
							seed /* a seed to generate random numbers */);


	cout << (boost::format("Calculating rejection frequencies for T=%d, num_samples=%d, and num_boots=%d") %T %num_samples %num_boots).str() << endl;
	pwr_out << (boost::format("Calculating rejection frequencies for T=%d, num_samples=%d, and num_boots=%d") %T %num_samples %num_boots).str() << endl;
	pwr_out << "Distribution of the errors = " << err_dist << endl;
	pwr_out << "A = " << A << endl;
	pwr_out << "B = " << B << endl;
	pwr_out << "C = " << C << endl;
	pwr_out << "mu = " << mu << endl;
	pwr_out << "error distribution = " << err_dist << endl;
	pwr_out << "nu = " << nu << endl;


	int i = 0, asymp_REJ1 = 0, asymp_REJ5 = 0, asymp_REJ10 = 0;
	int boot_REJ1 = 0, boot_REJ5 = 0, boot_REJ10 = 0;
	int sn_REJ1 = 0, sn_REJ5 = 0, sn_REJ10 = 0;
	int sn2_REJ1 = 0, sn2_REJ5 = 0, sn2_REJ10 = 0;
	int skip = 0;

	double t_pvalue = 0., boot_t_pvalue = 0., sn_stat = 0.;
	matrix<double> Xi(T,1), Yi(T,1);

	int num_bws = bws.nr(), num_blocks = Ls.nr();
	matrix<double> asymp_REJF1(num_bws,1), asymp_REJF5(num_bws,1), asymp_REJF10(num_bws,1);
	matrix<double> boot_REJF1(num_bws, num_blocks), boot_REJF5(num_bws, num_blocks), boot_REJF10(num_bws, num_blocks);
	double sn_REJF1, sn_REJF5, sn_REJF10;
	double sn2_REJF1, sn2_REJF5, sn2_REJF10;

	pwr_out << std::fixed << std::setprecision(5);

//	pwr_out << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;
//	cout << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;


	/* Compute Student's t-test p-values */
	pwr_out << "k, t_pvalue" << endl;
	cout << "k, t_pvalue" << endl;
	for (int k = 0; k < num_bws; ++k) {
		asymp_REJ1 = 0; //reset rejection frequencies
		asymp_REJ5 = 0;
		asymp_REJ10 = 0;
		for (i = 0; i < num_samples; ++i) {
			try {
				Xi = colm(X,i);
				Yi = colm(Y,i);

					//calculate Student's t-test p-value of the difference between two Sharpe ratios
				t_pvalue = TestSharpeR::t_pvalue_diff_sharpes<kernel>(Xi, /* excess returns on first asset */
																	  Yi, /* excess returns on second asset */
																	  bws(k) /* kernel bandwidth */);

				if (t_pvalue <= 0.01)  ++asymp_REJ1;//using 1%-level of significance
				if (t_pvalue <= 0.05)  ++asymp_REJ5;//using 5%-level of significance
				if (t_pvalue <= 0.10)  ++asymp_REJ10;//using 10%-level of significance

				pwr_out << bws(k) << " , " << t_pvalue << endl;
//				cout << bws(k) << " , " << t_pvalue << endl;
			}
			catch(...) { //catch all exceptions
					cerr << "Power::power_f: An exception occurs while calculating t_pvalue!" << endl;
					++skip;
			}
		}
		asymp_REJF1(k) = ((double) asymp_REJ1/(num_samples-skip) );
		asymp_REJF5(k) = ((double) asymp_REJ5/(num_samples-skip) );
		asymp_REJF10(k) = ((double) asymp_REJ10/(num_samples-skip) );

		pwr_out << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
		pwr_out << bws(k) << " , "  << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
//		cout << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
//		cout << bws(k) << " , " << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
	}


	/* Calculate Student's t-test bootstrap p-values */
	pwr_out << "k, h, boot_t_pvalue" << endl;
	cout << "k, h, boot_t_pvalue" << endl;
	for (int k = 0; k < num_bws; ++k) {
		for (int h = 0; h < num_blocks; ++h) {
//			pwr_out << (boost::format("bw = %f and block size = %d") %bw %L).str() << endl;
			boot_REJ1 = 0; //reset rejection frequencies
			boot_REJ5 = 0;
			boot_REJ10 = 0;
			skip = 0;
			#pragma omp parallel for default(shared) reduction(+:skip,boot_REJ1,boot_REJ5,boot_REJ10) \
																			schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,boot_t_pvalue)
			for (i = 0; i < num_samples; ++i) {
				try {
					Xi = colm(X,i);
					Yi = colm(Y,i);

					/* Calculate Student's t-test bootstrap p-value of the difference between two Sharpe ratios */
					boot_t_pvalue = TestSharpeR::t_boot_pvalue_diff_sharpes<kernel>(Xi, /* excess returns on first asset */
																					Yi, /* excess returns on second asset */
																					bws(k), /* kernel bandwidth */
																					Ls(h), /* block size */
																					num_boots, /* number of bootstrap repetitions */
																					seed /* a seed to generate random numbers */);

					#pragma omp critical
					{
						pwr_out << bws(k) << " , " << Ls(h) << " , " <<  boot_t_pvalue <<  endl;
//						cout << bws(k) << " , " << Ls(h) << " , " <<  boot_t_pvalue <<  endl;
					}

					if (boot_t_pvalue <= 0.01)  ++boot_REJ1;//using 1%-level of significance
					if (boot_t_pvalue <= 0.05)  ++boot_REJ5;//using 5%-level of significance
					if (boot_t_pvalue <= 0.10)  ++boot_REJ10;//using 10%-level of significance

				}
				catch(...) { //catch all exceptions
					cerr << "Power::power_f: An exception occurs while calculating boot_t_pvalue!" << endl;
					++skip;
				}
			}

			boot_REJF1(k,h) = ((double) boot_REJ1/(num_samples-skip) );
			boot_REJF5(k,h) = ((double) boot_REJ5/(num_samples-skip) );
			boot_REJF10(k,h) = ((double) boot_REJ10/(num_samples-skip) );

			pwr_out << "1%, 5%, and 10%-level rejection frequencies for the bootstrap Student's t-test" << endl;
			pwr_out << bws(k) << " , " << Ls(h) << " , " << boot_REJF1(k,h) << " , " << boot_REJF5(k,h) << " , " << boot_REJF10(k,h) << endl;
//			cout << "1%, 5%, and 10%-level rejection frequencies for the bootstrap Student's t-test" << endl;
//			cout << bws(k) << " , " << Ls(h) << " , " << boot_REJF1(k,h) << " , " << boot_REJF5(k,h) << " , " << boot_REJF10(k,h) << endl;
		}
	}

	/* Calculate the self-normalized test statistics */
	pwr_out << "sn_stat" << endl;
	cout << "sn_stat" << endl;
	skip = 0;
	for (i = 0; i < num_samples; ++i) {
		try {
			Xi = colm(X,i);
			Yi = colm(Y,i);

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn_sharpe_stat(Xi,Yi);

			pwr_out << sn_stat << endl;
//			cout << sn_stat << endl;

			//using the upper critical values tabulated in Table 1 of Lobato (2001)
			if (pow(sn_stat, 2.) >= 99.76) ++sn_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= 45.40) ++sn_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= 28.31) ++sn_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn_REJF1 = ((double) sn_REJ1/(num_samples-skip) );
	sn_REJF5 = ((double) sn_REJ5/(num_samples-skip) );
	sn_REJF10 = ((double) sn_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
//	cout << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;


	/* Calculate Volgushev and Shao's (2014) self-normalized test statistics */
	pwr_out << "sn2_stat" << endl;
	cout << "sn2_stat" << endl;
	skip = 0;
	for (i = 0; i < num_samples; ++i) {
		try {
			Xi = colm(X,i);
			Yi = colm(Y,i);

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn2_sharpe_stat(Xi,Yi);

			pwr_out << sn_stat << endl;
//			cout << sn_stat << endl;

			//using the upper critical values tabulated in Table 2 of Shao (2015)
			if (pow(sn_stat, 2.) >= 139.73) ++sn2_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= 68.41) ++sn2_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= 44.46) ++sn2_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn2_stat!" << endl;
			++skip;
		}
	}

	sn2_REJF1 = ((double) sn2_REJ1/(num_samples-skip) );
	sn2_REJF5 = ((double) sn2_REJ5/(num_samples-skip) );
	sn2_REJF10 = ((double) sn2_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for Volgushev and Shao's (2014) self-normalized t-test" << endl;
	pwr_out << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for Volgushev and Shao's (2014) self-normalized t-test" << endl;
//	cout << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;

	return {asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10};
}


/* Calculate empirical critical values of Student's t-test and the self-normalized t-test for the null hypothesis of equality between two Sharpe ratios */
template</* a HAC kernel */
		double kernel(double),
		/* a data generating process */
		std::pair<matrix<double>, matrix<double>> gen_data(const int, /* number of random samples */
															const int, /* sample size */
															const double, /* parameter of the error distribution */
															const matrix<double> &, /* means */
															const matrix<double> &, /* ARCH coefficients */
															const matrix<double> &, /* AR coefficients */
															const matrix<double> &, /* intercepts */
															const string,          /* distribution of the errors */
															unsigned long          /* a seed to generate random numbers */)>
std::tuple<matrix<double>, matrix<double>, matrix<double>,
			double, double, double,
			double, double, double> Power::critical_vals(const int num_samples, /* number of random samples */
														const int T, /* sample size */
														const matrix<double> &bws, /* kernel bandwidths */
														const double nu, /* parameter of the error distribution */
														const matrix<double> &mu, /* means */
														const matrix<double> &A, /* ARCH coefficients */
														const matrix<double> &B, /* AR coefficients */
														const matrix<double> &C, /* intercepts */
														const string err_dist, /* distribution of the errors */
														ofstream &pwr_out, /* output stream */
														unsigned long seed /* a seed to generate random numbers */) {

	ASSERT_(mu(0) == mu(1)); //check if the means are equal

	/* Generate many random samples */
    auto [X, Y] = gen_data(num_samples, /* number of random samples */
							T, /* sample size */
							nu, /* parameter of the error distribution */
							mu, /* means */
							A, /* ARCH coefficients */
							B, /* AR coefficients */
							C, /* intercepts */
							err_dist, /* distribution of the errors */
							seed /* a seed to generate random numbers */);


	cout << (boost::format("Calculating critical values for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << (boost::format("Calculating critical values for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << "Distribution of the errors = " << err_dist << endl;
	pwr_out << "A = " << A << endl;
	pwr_out << "B = " << B << endl;
	pwr_out << "C = " << C << endl;
	pwr_out << "mu = " << mu << endl;
	pwr_out << "error distribution = " << err_dist << endl;
	pwr_out << "nu = " << nu << endl;


	int i = 0;

	matrix<double> t_stats(num_samples,1), sn_stats(num_samples,1), sn2_stats(num_samples,1);
	matrix<double> Xi(T,1), Yi(T,1);

	matrix<double> t_cv1(bws.nr(),1), t_cv5(bws.nr(),1), t_cv10(bws.nr(),1);
	double sn_cv1, sn_cv5, sn_cv10;
	double sn2_cv1, sn2_cv5, sn2_cv10;

	pwr_out << std::fixed << std::setprecision(5);

	/* Calculate Student t-statistics */
	pwr_out << "k, t_stat" << endl;
	for (int k = 0; k < bws.nr(); ++k) {
		#pragma omp parallel for default(shared) schedule(static,CHUNK) private(i) firstprivate(Xi,Yi)
		for (i = 0; i < num_samples; ++i) {
			try {
				Xi = colm(X,i);
				Yi = colm(Y,i);

				//calculate Student's t-statistic of the difference between two Sharpe ratios
				t_stats(i) = TestSharpeR::t_stat_diff_sharpes<kernel>(Xi, /* excess returns on first asset */
																	  Yi, /* excess returns on second asset */
																	  bws(k) /* kernel bandwidth */);

				#pragma omp critical
				{
					pwr_out << bws(k) << " , " << t_stats(i) << endl;
				}
			}
			catch(...) { //catch all exceptions
				cerr << "Power::critical_vals: An exception occurs while calculating t_stat!" << endl;
			}
		}
		t_cv1(k) = find_upper_quantile(t_stats, 0.01);
		t_cv5(k) = find_upper_quantile(t_stats, 0.05);
		t_cv10(k) = find_upper_quantile(t_stats, 0.10);
	}

	pwr_out << "sn_stat" << endl;
	for (i = 0; i < num_samples; ++i) {
		try {
			Xi = colm(X,i);
			Yi = colm(Y,i);

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stats(i) = SnTestSharpeR::sn_sharpe_stat(Xi,Yi);

			pwr_out << sn_stats(i) << endl;
		}
		catch(...) { //catch all exceptions
			cerr << "Power::critical_vals: An exception occurs while calculating sn_stats!" << endl;
		}
	}

	sn_cv1 = find_upper_quantile(pointwise_multiply(sn_stats, sn_stats), 0.01);
	sn_cv5 = find_upper_quantile(pointwise_multiply(sn_stats, sn_stats), 0.05);
	sn_cv10 = find_upper_quantile(pointwise_multiply(sn_stats, sn_stats), 0.10);

	pwr_out << "sn2_stat" << endl;
	for (i = 0; i < num_samples; ++i) {
		try {
			Xi = colm(X,i);
			Yi = colm(Y,i);

			/* Calculate Volgushev and Shao's (2014) self-normalized t-statistic of the difference between two Sharpe ratios */
			sn2_stats(i) = SnTestSharpeR::sn2_sharpe_stat(Xi,Yi);

			pwr_out << sn2_stats(i) << endl;
		}
		catch(...) { //catch all exceptions
			cerr << "Power::critical_vals: An exception occurs while calculating sn2_stats!" << endl;
		}
	}

	sn2_cv1 = find_upper_quantile(pointwise_multiply(sn2_stats, sn2_stats), 0.01);
	sn2_cv5 = find_upper_quantile(pointwise_multiply(sn2_stats, sn2_stats), 0.05);
	sn2_cv10 = find_upper_quantile(pointwise_multiply(sn2_stats, sn2_stats), 0.10);

	return {t_cv1, t_cv5, t_cv10, sn_cv1, sn_cv5, sn_cv10, sn2_cv1, sn2_cv5, sn2_cv10};
}

/* Calculate empirical rejection rates using the empirical critical values of Student's t-test and the self-normalized t-tests
for the null hypothesis of equality between two Sharpe ratios */
template</* a HAC kernel */
		double kernel(double),
		/* a data generating process */
		std::pair<matrix<double>, matrix<double>> gen_data(const int, /* number of random samples */
															const int, /* sample size */
															const double, /* parameter of the error distribution */
															const matrix<double> &, /* means */
															const matrix<double> &, /* ARCH coefficients */
															const matrix<double> &, /* AR coefficients */
															const matrix<double> &, /* intercepts */
															const string,          /* distribution of the errors */
															unsigned long          /* a seed to generate random numbers */)>
std::tuple<matrix<double>, matrix<double>, matrix<double>,
			double, double, double,
			double, double, double> Power::epower_f(const int num_samples, /* number of random samples */
													const int T, /* sample size */
													const matrix<double> &bws, /* kernel bandwidths */
													const double nu, /* parameter of the error distribution */
													const matrix<double> &t_cv1, /* 1%-level critical values of Student's t-test */
													const matrix<double> &t_cv5, /* 5%-level critical values of Student's t-test */
													const matrix<double> &t_cv10, /* 10%-level critical values of Student's t-test */
													const double sn_cv1, /* 1%-level critical value of the self-normalized test */
													const double sn_cv5, /* 5%-level critical value of the self-normalized test */
													const double sn_cv10, /* 10%-level critical value of the self-normalized test */
													const double sn2_cv1, /* 1%-level critical value of Volgushev and Shao's (2014) self-normalized test */
													const double sn2_cv5, /* 5%-level critical value of Volgushev and Shao's (2014) self-normalized test */
													const double sn2_cv10, /* 10%-level critical value of Volgushev and Shao's (2014) self-normalized test */
													const matrix<double> &mu, /* means */
													const matrix<double> &A, /* ARCH coefficients */
													const matrix<double> &B, /* AR coefficients */
													const matrix<double> &C, /* intercepts */
													const string err_dist, /* distribution of the errors */
													ofstream &pwr_out, /* output stream */
													unsigned long seed /* a seed to generate random numbers */) {

	ASSERT_(mu(0) != mu(1));

	/* Generate many random samples */
    auto [X, Y] = gen_data(num_samples, /* number of random samples */
							T, /* sample size */
							nu, /* parameter of the error distribution */
							mu, /* means */
							A, /* ARCH coefficients */
							B, /* AR coefficients */
							C, /* intercepts */
							err_dist, /* distribution of the errors */
							seed /* a seed to generate random numbers */);


	cout << (boost::format("Calculating rejection frequencies for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << (boost::format("Calculating rejection frequencies for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << "Distribution of the errors = " << err_dist << endl;
	pwr_out << "A = " << A << endl;
	pwr_out << "B = " << B << endl;
	pwr_out << "C = " << C << endl;
	pwr_out << "mu = " << mu << endl;
	pwr_out << "error distribution = " << err_dist << endl;
	pwr_out << "nu = " << nu << endl;


	int i = 0, asymp_REJ1 = 0, asymp_REJ5 = 0, asymp_REJ10 = 0;
	int sn_REJ1 = 0, sn_REJ5 = 0, sn_REJ10 = 0;
	int sn2_REJ1 = 0, sn2_REJ5 = 0, sn2_REJ10 = 0;
	int skip = 0;

	double t_stat = 0., sn_stat = 0.;
	matrix<double> Xi(T,1), Yi(T,1);

	int num_bws = bws.nr();
	matrix<double> asymp_REJF1(num_bws,1), asymp_REJF5(num_bws,1), asymp_REJF10(num_bws,1);
	double sn_REJF1, sn_REJF5, sn_REJF10;
	double sn2_REJF1, sn2_REJF5, sn2_REJF10;

	pwr_out << std::fixed << std::setprecision(5);

//	pwr_out << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;
//	cout << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;


	/* Compute Student's t-test p-values */
	pwr_out << "k, t_pvalue" << endl;
	cout << "k, t_pvalue" << endl;
	for (int k = 0; k < num_bws; ++k) {
		asymp_REJ1 = 0; //reset rejection frequencies
		asymp_REJ5 = 0;
		asymp_REJ10 = 0;
		for (i = 0; i < num_samples; ++i) {
			try {
				Xi = colm(X,i);
				Yi = colm(Y,i);

				//calculate Student's t-statistic of the difference between two Sharpe ratios
				t_stat = TestSharpeR::t_stat_diff_sharpes<kernel>(Xi, /* excess returns on first asset */
																  Yi, /* excess returns on second asset */
																  bws(k) /* kernel bandwidth */);

				if (fabs(t_stat) >= t_cv1(k))  ++asymp_REJ1;//using 1%-level of significance
				if (fabs(t_stat) >= t_cv5(k))  ++asymp_REJ5;//using 5%-level of significance
				if (fabs(t_stat) >= t_cv10(k))  ++asymp_REJ10;//using 10%-level of significance

				pwr_out << bws(k) << " , " << t_stat << endl;
//				cout << bws(k) << " , " << t_pvalue << endl;
			}
			catch(...) { //catch all exceptions
					cerr << "Power::power_f: An exception occurs while calculating t_pvalue!" << endl;
					++skip;
			}
		}
		asymp_REJF1(k) = ((double) asymp_REJ1/(num_samples-skip) );
		asymp_REJF5(k) = ((double) asymp_REJ5/(num_samples-skip) );
		asymp_REJF10(k) = ((double) asymp_REJ10/(num_samples-skip) );

		pwr_out << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
		pwr_out << bws(k) << " , "  << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
//		cout << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
//		cout << bws(k) << " , " << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
	}


	/* Calculate the self-normalized test statistics */
	pwr_out << "sn_stat" << endl;
	cout << "sn_stat" << endl;
	skip = 0;
	for (i = 0; i < num_samples; ++i) {
		try {
			Xi = colm(X,i);
			Yi = colm(Y,i);

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn_sharpe_stat(Xi,Yi);

			pwr_out << sn_stat << endl;
//			cout << sn_stat << endl;

			//using the empirical upper critical values
			if (pow(sn_stat, 2.) >= sn_cv1) ++sn_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= sn_cv5) ++sn_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= sn_cv10) ++sn_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn_REJF1 = ((double) sn_REJ1/(num_samples-skip) );
	sn_REJF5 = ((double) sn_REJ5/(num_samples-skip) );
	sn_REJF10 = ((double) sn_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
//	cout << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;


	/* Calculate Volgushev and Shao's (2014) self-normalized test statistics */
	pwr_out << "sn2_stat" << endl;
	cout << "sn2_stat" << endl;
	skip = 0;
	for (i = 0; i < num_samples; ++i) {
		try {
			Xi = colm(X,i);
			Yi = colm(Y,i);

			/* Calculate Volgushev and Shao's (2014) self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn2_sharpe_stat(Xi,Yi);

			pwr_out << sn_stat << endl;
//			cout << sn_stat << endl;

			//using the empirical upper critical values
			if (pow(sn_stat, 2.) >= sn2_cv1) ++sn2_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= sn2_cv5) ++sn2_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= sn2_cv10) ++sn2_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn2_REJF1 = ((double) sn2_REJ1/(num_samples-skip) );
	sn2_REJF5 = ((double) sn2_REJ5/(num_samples-skip) );
	sn2_REJF10 = ((double) sn2_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
//	cout << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;

	return {asymp_REJF1, asymp_REJF5, asymp_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10};
}




/* Calculate empirical rejection frequencies of Student's t-test, the bootstrap t-test, and the self-normalized t-tests for the null hypothesis of
equality between two maximum squared Sharpe ratios that are attainable from two overlapping sets of factors */
template</* a HAC kernel */
		double kernel(double),
		/* a trivariate data generating process */
		Samples gen_data(const int, /* number of random samples */
						const int, /* sample size */
						const double, /* parameter of the error distribution */
						const matrix<double> &, /* means */
						const matrix<double> &, /* ARCH coefficients */
						const matrix<double> &, /* AR coefficients */
						const matrix<double> &, /* unconditional covariance matrix */
						const string,          /* distribution of the errors */
						unsigned long          /* a seed to generate random numbers */)>
std::tuple<matrix<double>, matrix<double>, matrix<double>,
			matrix<double>, matrix<double>, matrix<double>,
			double, double, double,
			double, double, double> Power::power_fovl(const int num_samples, /* number of random samples */
														const int T, /* sample size */
														const matrix<double> &bws, /* kernel bandwidths */
														const matrix<int> &Ls, /* block sizes */
														const int num_boots, /* number of bootstrap repetitions */
														const double nu, /* parameter of the error distribution */
														const matrix<double> &mu, /* means */
														const matrix<double> &A, /* ARCH coefficients */
														const matrix<double> &B, /* AR coefficients */
														const matrix<double> &Omega, /* unconditional covariance matrix */
														const string err_dist, /* distribution of the errors */
														const bool fovl, /* True if there are overlapping factors */
														ofstream &pwr_out, /* output stream */
														unsigned long seed /* a seed to generate random numbers */) {

	/* Generate many random samples */
    auto X = gen_data(num_samples, /* number of random samples */
						T, /* sample size */
						nu, /* parameter of the error distribution */
						mu, /* means */
						A, /* ARCH coefficients */
						B, /* AR coefficients */
						Omega, /* unconditional covariance matrix */
						err_dist, /* distribution of the errors */
						seed /* a seed to generate random numbers */);


	cout << (boost::format("Calculating rejection frequencies for T=%d, num_samples=%d, and num_boots=%d") %T %num_samples %num_boots).str() << endl;
	pwr_out << (boost::format("Calculating rejection frequencies for T=%d, num_samples=%d, and num_boots=%d") %T %num_samples %num_boots).str() << endl;
	pwr_out << "Distribution of the errors = " << err_dist << endl;
	pwr_out << "A = " << A << endl;
	pwr_out << "B = " << B << endl;
	pwr_out << "Omega = " << Omega << endl;
	pwr_out << "mu = " << mu << endl;
	pwr_out << "error distribution = " << err_dist << endl;
	pwr_out << "nu = " << nu << endl;


	int i = 0, asymp_REJ1 = 0, asymp_REJ5 = 0, asymp_REJ10 = 0;
	int boot_REJ1 = 0, boot_REJ5 = 0, boot_REJ10 = 0;
	int sn_REJ1 = 0, sn_REJ5 = 0, sn_REJ10 = 0;
	int sn2_REJ1 = 0, sn2_REJ5 = 0, sn2_REJ10 = 0;
	int skip = 0;

	double t_pvalue = 0., boot_t_pvalue = 0., sn_stat = 0.;
	matrix<double> Xi(T,2), Yi(T,2);

	int num_bws = bws.nr(), num_blocks = Ls.nr();
	matrix<double> asymp_REJF1(num_bws,1), asymp_REJF5(num_bws,1), asymp_REJF10(num_bws,1);
	matrix<double> boot_REJF1(num_bws, num_blocks), boot_REJF5(num_bws, num_blocks), boot_REJF10(num_bws, num_blocks);
	double sn_REJF1, sn_REJF5, sn_REJF10;
	double sn2_REJF1, sn2_REJF5, sn2_REJF10;

	pwr_out << std::fixed << std::setprecision(5);

	/* Calculate Student's t-test p-values of the difference between two maximum squared Sharpe ratios */
	pwr_out << "k, t_pvalue" << endl;
//	cout << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;
	for (int k = 0; k < bws.nr(); ++k) {
//			pwr_out << (boost::format("bw = %f and block size = %d") %bw %L).str() << endl;
		asymp_REJ1 = 0; //reset rejection frequencies
		asymp_REJ5 = 0;
		asymp_REJ10 = 0;
		skip = 0;
		#pragma omp parallel for default(shared) reduction(+:skip,asymp_REJ1,asymp_REJ5,asymp_REJ10) \
									schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,t_pvalue)
		for (i = 0; i < num_samples; ++i) {
			try {
				if (fovl == true) {
					Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
					Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
				}
				else {
					Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
					Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
				}

				//calculate Student's t-test p-value of the difference between two Sharpe ratios
				t_pvalue = TestSharpeR::t_pvalue_diff_max_sq_sharpes<kernel>(Xi, /* a T by N1 matrix of factors */
																			 Yi, /* a T by N2 matrix of factors */
																			 bws(k) /* kernel bandwidth */);

				#pragma omp critical
				{
					pwr_out << bws(k) << " , " << t_pvalue  << endl;
//						cout << bws(k) << " , " << t_pvalue  << endl;
				}

				if (t_pvalue <= 0.01)  ++asymp_REJ1;//using 1%-level of significance
				if (t_pvalue <= 0.05)  ++asymp_REJ5;//using 5%-level of significance
				if (t_pvalue <= 0.10)  ++asymp_REJ10;//using 10%-level of significance
			}
			catch(...) { //catch all exceptions
				cerr << "Power::power_f: An exception occurs!" << endl;
				++skip;
			}
		}
		asymp_REJF1(k) = ((double) asymp_REJ1/(num_samples-skip) );
		asymp_REJF5(k) = ((double) asymp_REJ5/(num_samples-skip) );
		asymp_REJF10(k) = ((double) asymp_REJ10/(num_samples-skip) );

		pwr_out << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
		pwr_out << bws(k) << " , "  << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
//		cout << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
//		cout << bws(k) << " , " << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
	}


	/* Calculate Student's t-test bootstrap p-values of the difference between two maximum squared Sharpe ratios */
	pwr_out << "k, h, boot_t_pvalue" << endl;
//	cout << "k, h, boot_t_pvalue" << endl;
	for (int k = 0; k < num_bws; ++k) {
		for (int h = 0; h < num_blocks; ++h) {
//			pwr_out << (boost::format("bw = %f and block size = %d") %bw %L).str() << endl;
			boot_REJ1 = 0; //reset rejection frequencies
			boot_REJ5 = 0;
			boot_REJ10 = 0;
			skip = 0;
			#pragma omp parallel for default(shared) reduction(+:skip,boot_REJ1,boot_REJ5,boot_REJ10) \
																schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,boot_t_pvalue)
			for (i = 0; i < num_samples; ++i) {
				try {
					if (fovl == true) {
						Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
						Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
					}
					else {
						Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
						Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
					}

					/* Calculate Student's t-test bootstrap p-value of the difference between two Sharpe ratios */
					boot_t_pvalue = TestSharpeR::t_boot_pvalue_diff_max_sq_sharpes<kernel>(Xi, /* a T by N1 matrix of factors */
																							Yi, /* a T by N2 matrix of factors */
																							bws(k), /* kernel bandwidth */
																							Ls(h), /* block size */
																							num_boots, /* number of bootstrap repetitions */
																							seed /* a seed to generate random numbers */);



					#pragma omp critical
					{
						pwr_out << bws(k) << " , " << Ls(h) << " , " << boot_t_pvalue <<  endl;
//						cout << bws(k) << " , " << Ls(h) << " , " << boot_t_pvalue <<  endl;
					}

					if (boot_t_pvalue <= 0.01)  ++boot_REJ1;//using 1%-level of significance
					if (boot_t_pvalue <= 0.05)  ++boot_REJ5;//using 5%-level of significance
					if (boot_t_pvalue <= 0.10)  ++boot_REJ10;//using 10%-level of significance

				}
				catch(...) { //catch all exceptions
					cerr << "Power::power_f: An exception occurs!" << endl;
					++skip;
				}
			}
			boot_REJF1(k,h) = ((double) boot_REJ1/(num_samples-skip) );
			boot_REJF5(k,h) = ((double) boot_REJ5/(num_samples-skip) );
			boot_REJF10(k,h) = ((double) boot_REJ10/(num_samples-skip) );

			pwr_out << "1%, 5%, and 10%-level rejection frequencies for the bootstrap Student's t-test" << endl;
			pwr_out << k << " , " << h << " , " << boot_REJF1(k,h) << " , " << boot_REJF5(k,h) << " , " << boot_REJF10(k,h) << endl;
//			cout << "1%, 5%, and 10%-level rejection frequencies for the bootstrap Student's t-test" << endl;
//			cout << k << " , " << h << " , " << boot_REJF1(k,h) << " , " << boot_REJF5(k,h) << " , " << boot_REJF10(k,h) << endl;
		}
	}

	/* Calculate the self-normalized test statistics of the difference between two maximum squared Sharpe ratios */
	pwr_out << "sn_stat" << endl;
	cout << "sn_stat" << endl;
	skip = 0;
	#pragma omp parallel for default(shared) reduction(+:skip,sn_REJ1,sn_REJ5,sn_REJ10) \
									schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,sn_stat)
	for (i = 0; i < num_samples; ++i) {
		try {
			if (fovl == true) {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
			}
			else {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
			}

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn_max_sq_sharpe_stat(Xi,Yi);

			#pragma omp critical
			{
				pwr_out << sn_stat << endl;
	//			cout << sn_stat << endl;
			}

			//using the upper critical values tabulated in Table 1 of Lobato (2001)
			if (pow(sn_stat, 2.) >= 99.76) ++sn_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= 45.40) ++sn_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= 28.31) ++sn_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn_REJF1 = ((double) sn_REJ1/(num_samples-skip) );
	sn_REJF5 = ((double) sn_REJ5/(num_samples-skip) );
	sn_REJF10 = ((double) sn_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
//	cout << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;


	/* Calculate Volgushev and Shao's (2014) self-normalized test statistics of the difference between two maximum squared Sharpe ratios */
	pwr_out << "sn2_stat" << endl;
	cout << "sn2_stat" << endl;
	skip = 0;
	#pragma omp parallel for default(shared) reduction(+:skip,sn2_REJ1,sn2_REJ5,sn2_REJ10) \
									schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,sn_stat)
	for (i = 0; i < num_samples; ++i) {
		try {
			if (fovl == true) {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
			}
			else {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
			}

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn2_max_sq_sharpe_stat(Xi,Yi);

			#pragma omp critical
			{
				pwr_out << sn_stat << endl;
	//			cout << sn_stat << endl;
			}

			//using the upper critical values tabulated in Table 2 of Shao (2015)
			if (pow(sn_stat, 2.) >= 139.73) ++sn2_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= 68.41) ++sn2_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= 44.46) ++sn2_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn2_REJF1 = ((double) sn2_REJ1/(num_samples-skip) );
	sn2_REJF5 = ((double) sn2_REJ5/(num_samples-skip) );
	sn2_REJF10 = ((double) sn2_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for Volgushev and Shao's (2014) self-normalized t-test" << endl;
//	cout << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;


	return {asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10};
}


/* Calculate empirical critical values of Student's t-test and the self-normalized t-tests for the null hypothesis of equality between
two maximum squared Sharpe ratios that are attainable from two overlapping sets of factors */
template</* a HAC kernel */
		double kernel(double),
		/* a trivariate data generating process */
		Samples gen_data(const int, /* number of random samples */
						const int, /* sample size */
						const double, /* parameter of the error distribution */
						const matrix<double> &, /* means */
						const matrix<double> &, /* ARCH coefficients */
						const matrix<double> &, /* AR coefficients */
						const matrix<double> &, /* unconditional covariance matrix */
						const string,          /* distribution of the errors */
						unsigned long          /* a seed to generate random numbers */)>
std::tuple<matrix<double>, matrix<double>, matrix<double>,
			double, double, double,
			double, double, double> Power::critical_vals_fovl(const int num_samples, /* number of random samples */
															const int T, /* sample size */
															const matrix<double> &bws, /* kernel bandwidths */
															const double nu, /* parameter of the error distribution */
															const matrix<double> &mu, /* means */
															const matrix<double> &A, /* ARCH coefficients */
															const matrix<double> &B, /* AR coefficients */
															const matrix<double> &Omega, /* unconditional covariance matrix */
															const string err_dist, /* distribution of the errors */
															const bool fovl, /* True if there are overlapping factors */
															ofstream &pwr_out, /* output stream */
															unsigned long seed /* a seed to generate random numbers */) {

	ASSERT_(mu(0) == mu(1) && mu(1) == mu(2) && mu(2) == mu(3)); //check if the means are equal

	/* Generate many random samples */
    auto X = gen_data(num_samples, /* number of random samples */
						T, /* sample size */
						nu, /* parameter of the error distribution */
						mu, /* means */
						A, /* ARCH coefficients */
						B, /* AR coefficients */
						Omega, /* unconditional covariance matrix */
						err_dist, /* distribution of the errors */
						seed /* a seed to generate random numbers */);


	cout << (boost::format("Calculating critical values for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << (boost::format("Calculating critical values for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << "Distribution of the errors = " << err_dist << endl;
	pwr_out << "A = " << A << endl;
	pwr_out << "B = " << B << endl;
	pwr_out << "Omega = " << Omega << endl;
	pwr_out << "mu = " << mu << endl;
	pwr_out << "error distribution = " << err_dist << endl;
	pwr_out << "nu = " << nu << endl;

	int i = 0;

	matrix<double> t_stats(num_samples,1), sn_stats(num_samples,1), sn2_stats(num_samples,1);
	matrix<double> Xi(T,1), Yi(T,1);

	matrix<double> t_cv1(bws.nr(),1), t_cv5(bws.nr(),1), t_cv10(bws.nr(),1);
	double sn_cv1, sn_cv5, sn_cv10;
	double sn2_cv1, sn2_cv5, sn2_cv10;

	pwr_out << std::fixed << std::setprecision(5);

	/* Calculate Student's t-statistic of the difference between two maximum squared Sharpe ratios */
	pwr_out << "k, t_stat" << endl;
//	cout << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;
	for (int k = 0; k < bws.nr(); ++k) {
		#pragma omp parallel for default(shared) schedule(static,CHUNK) private(i) firstprivate(Xi,Yi)
		for (i = 0; i < num_samples; ++i) {
			try {
				if (fovl == true) {
					Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
					Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
				}
				else {
					Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
					Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
				}

				//calculate Student's t-statistic of the difference between two Sharpe ratios
				t_stats(i) = TestSharpeR::t_stat_diff_max_sq_sharpes<kernel>(Xi, /* a T by N1 matrix of factors */
																			 Yi, /* a T by N2 matrix of factors */
																			 bws(k) /* kernel bandwidth */);

				#pragma omp critical
				{
					pwr_out << bws(k) << " , " << t_stats(i)  << endl;
//						cout << bws(k) << " , " << t_pvalue  << endl;
				}
			}
			catch(...) { //catch all exceptions
				cerr << "Power::critical_vals_fovl: An exception occurs while calculating t_stat!" << endl;
			}
		}
		t_cv1(k) = find_upper_quantile(t_stats, 0.01);
		t_cv5(k) = find_upper_quantile(t_stats, 0.05);
		t_cv10(k) = find_upper_quantile(t_stats, 0.10);
	}


	/* Calculate the self-normalized test statistics of the difference between two maximum squared Sharpe ratios */
	pwr_out << "sn_stat" << endl;
	cout << "sn_stat" << endl;
	#pragma omp parallel for default(shared) schedule(static,CHUNK) private(i) firstprivate(Xi,Yi)
	for (i = 0; i < num_samples; ++i) {
		try {
			if (fovl == true) {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
			}
			else {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
			}

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stats(i) = SnTestSharpeR::sn_max_sq_sharpe_stat(Xi,Yi);

			#pragma omp critical
			{
				pwr_out << sn_stats(i) << endl;
	//			cout << sn_stat << endl;
			}

		}
		catch(...) { //catch all exceptions
			cerr << "Power::critical_vals_fovl: An exception occurs while calculating sn_stat!" << endl;
		}
	}

	sn_cv1 = find_upper_quantile(pointwise_multiply(sn_stats, sn_stats), 0.01);
	sn_cv5 = find_upper_quantile(pointwise_multiply(sn_stats, sn_stats), 0.05);
	sn_cv10 = find_upper_quantile(pointwise_multiply(sn_stats, sn_stats), 0.10);


	/* Calculate Volgushev and Shao's (2014) self-normalized test statistics of the difference between two maximum squared Sharpe ratios */
	pwr_out << "sn2_stat" << endl;
	cout << "sn2_stat" << endl;
	#pragma omp parallel for default(shared) schedule(static,CHUNK) private(i) firstprivate(Xi,Yi)
	for (i = 0; i < num_samples; ++i) {
		try {
			if (fovl == true) {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
			}
			else {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
			}

			/* Calculate Volgushev and Shao's (2014) self-normalized t-statistic of the difference between two Sharpe ratios */
			sn2_stats(i) = SnTestSharpeR::sn2_max_sq_sharpe_stat(Xi,Yi);

			#pragma omp critical
			{
				pwr_out << sn2_stats(i) << endl;
	//			cout << sn_stat << endl;
			}

		}
		catch(...) { //catch all exceptions
			cerr << "Power::critical_vals_fovl: An exception occurs while calculating sn2_stat!" << endl;
		}
	}

	sn2_cv1 = find_upper_quantile(pointwise_multiply(sn2_stats, sn2_stats), 0.01);
	sn2_cv5 = find_upper_quantile(pointwise_multiply(sn2_stats, sn2_stats), 0.05);
	sn2_cv10 = find_upper_quantile(pointwise_multiply(sn2_stats, sn2_stats), 0.10);

	return {t_cv1, t_cv5, t_cv10, sn_cv1, sn_cv5, sn_cv10, sn2_cv1, sn2_cv5, sn2_cv10};
}


/* Calculate empirical rejection rates using the empirical critical values of Student's t-test and the self-normalized t-tests
for the null hypothesis of equality between two maximum squared Sharpe ratios that are attainable from two overlapping sets of factors */
template</* a HAC kernel */
		double kernel(double),
		/* a data generating process */
		Samples gen_data(const int, /* number of random samples */
						const int, /* sample size */
						const double, /* parameter of the error distribution */
						const matrix<double> &, /* means */
						const matrix<double> &, /* ARCH coefficients */
						const matrix<double> &, /* AR coefficients */
						const matrix<double> &, /* unconditional covariance matrix */
						const string,          /* distribution of the errors */
						unsigned long          /* a seed to generate random numbers */)>
std::tuple<matrix<double>, matrix<double>, matrix<double>,
			double, double, double,
			double, double, double> Power::epower_fovl(const int num_samples, /* number of random samples */
														const int T, /* sample size */
														const matrix<double> &bws, /* kernel bandwidths */
														const double nu, /* parameter of the error distribution */
														const matrix<double> &t_cv1, /* 1%-level critical values of Student's t-test */
														const matrix<double> &t_cv5, /* 5%-level critical values of Student's t-test */
														const matrix<double> &t_cv10, /* 10%-level critical values of Student's t-test */
														const double sn_cv1, /* 1%-level critical value of the self-normalized test */
														const double sn_cv5, /* 5%-level critical value of the self-normalized test */
														const double sn_cv10, /* 10%-level critical value of the self-normalized test */
														const double sn2_cv1, /* 1%-level critical value of Volgushev and Shao's (2014) self-normalized test */
														const double sn2_cv5, /* 5%-level critical value of Volgushev and Shao's (2014) self-normalized test */
														const double sn2_cv10, /* 10%-level critical value of Volgushev and Shao's (2014) self-normalized test */
														const matrix<double> &mu, /* means */
														const matrix<double> &A, /* ARCH coefficients */
														const matrix<double> &B, /* AR coefficients */
														const matrix<double> &Omega, /* unconditional covariance matrix */
														const string err_dist, /* distribution of the errors */
														const bool fovl, /* True if there are overlapping factors */
														ofstream &pwr_out, /* output stream */
														unsigned long seed /* a seed to generate random numbers */) {

	ASSERT_(mu(0) != mu(1) || mu(1) != mu(2));

	/* Generate many random samples */
    auto X = gen_data(num_samples, /* number of random samples */
						T, /* sample size */
						nu, /* parameter of the error distribution */
						mu, /* means */
						A, /* ARCH coefficients */
						B, /* AR coefficients */
						Omega, /* unconditional covariance matrix */
						err_dist, /* distribution of the errors */
						seed /* a seed to generate random numbers */);


	cout << (boost::format("Calculating rejection frequencies for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << (boost::format("Calculating rejection frequencies for T=%d and num_samples=%d") %T %num_samples).str() << endl;
	pwr_out << "Distribution of the errors = " << err_dist << endl;
	pwr_out << "A = " << A << endl;
	pwr_out << "B = " << B << endl;
	pwr_out << "Omega = " << Omega << endl;
	pwr_out << "mu = " << mu << endl;
	pwr_out << "error distribution = " << err_dist << endl;
	pwr_out << "nu = " << nu << endl;


	int i = 0, asymp_REJ1 = 0, asymp_REJ5 = 0, asymp_REJ10 = 0;
	int sn_REJ1 = 0, sn_REJ5 = 0, sn_REJ10 = 0;
	int sn2_REJ1 = 0, sn2_REJ5 = 0, sn2_REJ10 = 0;
	int skip = 0;

	double t_stat = 0., sn_stat = 0.;
	matrix<double> Xi(T,1), Yi(T,1);

	int num_bws = bws.nr();
	matrix<double> asymp_REJF1(num_bws,1), asymp_REJF5(num_bws,1), asymp_REJF10(num_bws,1);
	double sn_REJF1, sn_REJF5, sn_REJF10;
	double sn2_REJF1, sn2_REJF5, sn2_REJF10;

	pwr_out << std::fixed << std::setprecision(5);

//	pwr_out << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;
//	cout << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;


	/* Compute Student's t-tes p-values */
	pwr_out << "k, t_pvalue" << endl;
	cout << "k, t_pvalue" << endl;
	for (int k = 0; k < num_bws; ++k) {
		asymp_REJ1 = 0; //reset rejection frequencies
		asymp_REJ5 = 0;
		asymp_REJ10 = 0;
		#pragma omp parallel for default(shared) reduction(+:skip,asymp_REJ1,asymp_REJ5,asymp_REJ10) \
											schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,t_stat)
		for (i = 0; i < num_samples; ++i) {
			try {
				if (fovl == true) {
					Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
					Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
				}
				else {
					Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
					Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
				}

				//calculate Student's t-statistic of the difference between two Sharpe ratios
				t_stat = TestSharpeR::t_stat_diff_max_sq_sharpes<kernel>(Xi, /* a T by N1 matrix of factors */
																			 Yi, /* a T by N2 matrix of factors */
																			 bws(k) /* kernel bandwidth */);

				if (fabs(t_stat) >= t_cv1(k))  ++asymp_REJ1;//using 1%-level of significance
				if (fabs(t_stat) >= t_cv5(k))  ++asymp_REJ5;//using 5%-level of significance
				if (fabs(t_stat) >= t_cv10(k))  ++asymp_REJ10;//using 10%-level of significance

				pwr_out << bws(k) << " , " << t_stat << endl;
//				cout << bws(k) << " , " << t_pvalue << endl;
			}
			catch(...) { //catch all exceptions
					cerr << "Power::epower_fovl: An exception occurs while calculating t_pvalue!" << endl;
					++skip;
			}
		}
		asymp_REJF1(k) = ((double) asymp_REJ1/(num_samples-skip) );
		asymp_REJF5(k) = ((double) asymp_REJ5/(num_samples-skip) );
		asymp_REJF10(k) = ((double) asymp_REJ10/(num_samples-skip) );

		pwr_out << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
		pwr_out << bws(k) << " , "  << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
//		cout << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
//		cout << bws(k) << " , " << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
	}


	/* Calculate the self-normalized test statistics */
	pwr_out << "sn_stat" << endl;
	cout << "sn_stat" << endl;
	skip = 0;
	#pragma omp parallel for default(shared) reduction(+:skip,sn_REJ1,sn_REJ5,sn_REJ10) \
											schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,sn_stat)
	for (i = 0; i < num_samples; ++i) {
		try {
			if (fovl == true) {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
			}
			else {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
			}

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn_max_sq_sharpe_stat(Xi,Yi);

			pwr_out << sn_stat << endl;
//			cout << sn_stat << endl;

			//using the upper critical values tabulated in Table 1 of Lobato (2001)
			if (pow(sn_stat, 2.) >= sn_cv1) ++sn_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= sn_cv5) ++sn_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= sn_cv10) ++sn_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::epower_fovl: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn_REJF1 = ((double) sn_REJ1/(num_samples-skip) );
	sn_REJF5 = ((double) sn_REJ5/(num_samples-skip) );
	sn_REJF10 = ((double) sn_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
//	cout << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;


	/* Calculate Volgushev and Shao's (2014) self-normalized test statistics */
	pwr_out << "sn2_stat" << endl;
	cout << "sn2_stat" << endl;
	skip = 0;
	#pragma omp parallel for default(shared) reduction(+:skip,sn2_REJ1,sn2_REJ5,sn2_REJ10) \
											schedule(static,CHUNK) private(i) firstprivate(Xi,Yi,sn_stat)
	for (i = 0; i < num_samples; ++i) {
		try {
			if (fovl == true) {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(0,2,2)); /* a T by 2 matrix of factors */
			}
			else {
				Xi = colm(X[i], range(0,1)); /* a T by 2 matrix of factors */
				Yi = colm(X[i], range(2,3)); /* a T by 2 matrix of factors */
			}

			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn2_max_sq_sharpe_stat(Xi,Yi);

			pwr_out << sn_stat << endl;
//			cout << sn_stat << endl;

			//using the upper critical values tabulated in Table 1 of Lobato (2001)
			if (pow(sn_stat, 2.) >= sn2_cv1) ++sn2_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= sn2_cv5) ++sn2_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= sn2_cv10) ++sn2_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::epower_fovl: An exception occurs while calculating sn2_stat!" << endl;
			++skip;
		}
	}

	sn2_REJF1 = ((double) sn2_REJ1/(num_samples-skip) );
	sn2_REJF5 = ((double) sn2_REJ5/(num_samples-skip) );
	sn2_REJF10 = ((double) sn2_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for Volgushev and Shao's (2014) self-normalized t-test" << endl;
	pwr_out << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for Volgushev and Shao's (2014) self-normalized t-test" << endl;
//	cout << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;

	return {asymp_REJF1, asymp_REJF5, asymp_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10};
}


/* Calculate empirical rejection frequencies of Student's t-test, the bootstrap t-test, and the self-normalized t-tests for the null hypothesis of
equality between two maximum squared Sharpe ratios that are attainable from two sets of non-traded factors
[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
template</* a HAC kernel */
		double kernel(double),
		/* a trivariate data generating process */
		Samples gen_data(const int, /* number of random samples */
						const int, /* sample size */
						const double, /* parameter of the error distribution */
						const matrix<double> &, /* means */
						const matrix<double> &, /* ARCH coefficients */
						const matrix<double> &, /* AR coefficients */
						const matrix<double> &, /* unconditional covariance matrix */
						const string,          /* distribution of the errors */
						unsigned long          /* a seed to generate random numbers */)>
std::tuple<matrix<double>, matrix<double>, matrix<double>,
			matrix<double>, matrix<double>, matrix<double>,
			double, double, double,
			double, double, double> Power::power_fovl_mimicking(const int num_samples, /* number of random samples */
														const int T, /* sample size */
														const matrix<double> &bws, /* kernel bandwidths */
														const matrix<int> &Ls, /* block sizes */
														const int num_boots, /* number of bootstrap repetitions */
														const double nu, /* parameter of the error distribution */
														const matrix<double> &mu1, /* means of traded factors and basis asset returns */
														const matrix<double> &mu2, /* means of non-traded factors */
														const matrix<double> &A, /* ARCH coefficients */
														const matrix<double> &B, /* AR coefficients */
														const matrix<double> &C, /* Regression coefficients used to generate non-traded factors */
														const matrix<double> &Omega1, /* unconditional covariance matrx of the innovations used to generate traded factors */
														const matrix<double> &Omega2, /* covariance matrix of the innovations used to generate non-traded factors */
														const string err_dist, /* distribution of the errors */
														const bool fovl, /* True if there are overlapping factors */
														ofstream &pwr_out, /* output stream */
														unsigned long seed /* a seed to generate random numbers */) {

	/* Generate many random samples of traded factors and basis asset returns */
	auto X_all = gen_data(num_samples, /* number of random samples */
						T, /* sample size */
						nu, /* parameter of the error distribution */
						mu1, /* means */
						A, /* ARCH coefficients */
						B, /* AR coefficients */
						Omega1, /* unconditional covariance matrix */
						err_dist, /* distribution of the errors */
						seed /* a seed to generate random numbers */);

	Samples Z_x(num_samples, matrix<double>(T,2)), Z_y(num_samples, matrix<double>(T,2)); //random samples of traded factors and basis asset returns

	for (int i = 0; i < num_samples; ++i) {
		if (fovl == true) {
			Z_x[i] = colm(X_all[i], range(0,1)); /* a T by 2 matrix of factors */
			Z_y[i] = colm(X_all[i], range(0,2,2)); /* a T by 2 matrix of factors */
		}
		else {
			Z_x[i] = colm(X_all[i], range(0,1)); /* a T by 2 matrix of factors */
			Z_y[i] = colm(X_all[i], range(2,3)); /* a T by 2 matrix of factors */
		}
	}

	//generate random samples of non-traded factors from random samples of traded factors
	Samples X(num_samples, matrix<double>(T,mu2.nr())), Y(num_samples, matrix<double>(T,mu2.nr()));
	if (err_dist == "Student") {
		X = Dgp::gen_non_traded_factors<gsl_ran_tdist>(Z_x, /* random samples of traded factors and test assets */
															nu, /* parameter of the error distribution */
															mu2, /* means */
															C, /* Regression coefficients */
															Omega2, /* unconditional covariance matrix */
															err_dist, /* distribution of the errors */
															seed /* a seed to generate random numbers */);

		Y = Dgp::gen_non_traded_factors<gsl_ran_tdist>(Z_y, /* random samples of traded factors and test assets */
															nu, /* parameter of the error distribution */
															mu2, /* means */
															C, /* Regression coefficients */
															Omega2, /* unconditional covariance matrix */
															err_dist, /* distribution of the errors */
															seed /* a seed to generate random numbers */);
	}
	else {
		X = Dgp::gen_non_traded_factors<gsl_ran_gaussian>(Z_x, /* random samples of traded factors and test assets */
															nu, /* parameter of the error distribution */
															mu2, /* means */
															C, /* Regression coefficients */
															Omega2, /* unconditional covariance matrix */
															err_dist, /* distribution of the errors */
															seed /* a seed to generate random numbers */);

		Y = Dgp::gen_non_traded_factors<gsl_ran_gaussian>(Z_y, /* random samples of traded factors and test assets */
															nu, /* parameter of the error distribution */
															mu2, /* means */
															C, /* Regression coefficients */
															Omega2, /* unconditional covariance matrix */
															err_dist, /* distribution of the errors */
															seed /* a seed to generate random numbers */);
	}

	cout << (boost::format("Calculating rejection frequencies for T=%d, num_samples=%d, and num_boots=%d") %T %num_samples %num_boots).str() << endl;
	pwr_out << (boost::format("Calculating rejection frequencies for T=%d, num_samples=%d, and num_boots=%d") %T %num_samples %num_boots).str() << endl;
	pwr_out << "Distribution of the errors = " << err_dist << endl;
	pwr_out << "A = " << A << endl;
	pwr_out << "B = " << B << endl;
	pwr_out << "C = " << C << endl;
	pwr_out << "Omega1 = " << Omega1 << endl;
	pwr_out << "Omega2 = " << Omega2 << endl;
	pwr_out << "mu1 = " << mu1 << endl;
	pwr_out << "mu2 = " << mu2 << endl;
	pwr_out << "error distribution = " << err_dist << endl;
	pwr_out << "nu = " << nu << endl;


	int i = 0, asymp_REJ1 = 0, asymp_REJ5 = 0, asymp_REJ10 = 0;
	int boot_REJ1 = 0, boot_REJ5 = 0, boot_REJ10 = 0;
	int sn_REJ1 = 0, sn_REJ5 = 0, sn_REJ10 = 0;
	int sn2_REJ1 = 0, sn2_REJ5 = 0, sn2_REJ10 = 0;
	int skip = 0;

	double t_pvalue = 0., boot_t_pvalue = 0., sn_stat = 0.;

	int num_bws = bws.nr(), num_blocks = Ls.nr();
	matrix<double> asymp_REJF1(num_bws,1), asymp_REJF5(num_bws,1), asymp_REJF10(num_bws,1);
	matrix<double> boot_REJF1(num_bws, num_blocks), boot_REJF5(num_bws, num_blocks), boot_REJF10(num_bws, num_blocks);
	double sn_REJF1, sn_REJF5, sn_REJF10;
	double sn2_REJF1, sn2_REJF5, sn2_REJF10;

	pwr_out << std::fixed << std::setprecision(5);

	/* Calculate Student's t-test p-values of the difference between two maximum squared Sharpe ratios */
	pwr_out << "k, t_pvalue" << endl;
//	cout << "k, h, t_pvalue , boot_t_pvalue, sn_stat" << endl;
	for (int k = 0; k < bws.nr(); ++k) {
//			pwr_out << (boost::format("bw = %f and block size = %d") %bw %L).str() << endl;
		asymp_REJ1 = 0; //reset rejection frequencies
		asymp_REJ5 = 0;
		asymp_REJ10 = 0;
		skip = 0;
		#pragma omp parallel for default(shared) reduction(+:skip,asymp_REJ1,asymp_REJ5,asymp_REJ10) \
									schedule(static,CHUNK) private(i) firstprivate(t_pvalue)
		for (i = 0; i < num_samples; ++i) {
			try {
				//calculate Student's t-test p-value of the difference between two Sharpe ratios
				t_pvalue = TestSharpeR::t_pvalue_diff_max_sq_sharpes_mimicking<kernel>(X[i], /* a T by N1 matrix of non-traded factors */
																						Y[i], /* a T by N2 matrix of non-traded factors */
																						Z_x[i], /* a T by M1 matrix of traded factors and basis asset returns */
																						Z_y[i], /* a T by M2 matrix of traded factors and basis asset returns */
																						bws(k) /* kernel bandwidth */);
				#pragma omp critical
				{
					pwr_out << bws(k) << " , " << t_pvalue  << endl;
//					cout << bws(k) << " , " << t_pvalue  << endl;
				}

				if (t_pvalue <= 0.01)  ++asymp_REJ1;//using 1%-level of significance
				if (t_pvalue <= 0.05)  ++asymp_REJ5;//using 5%-level of significance
				if (t_pvalue <= 0.10)  ++asymp_REJ10;//using 10%-level of significance
			}
			catch(...) { //catch all exceptions
				cerr << "Power::power_f: An exception occurs!" << endl;
				++skip;
			}
		}
		asymp_REJF1(k) = ((double) asymp_REJ1/(num_samples-skip) );
		asymp_REJF5(k) = ((double) asymp_REJ5/(num_samples-skip) );
		asymp_REJF10(k) = ((double) asymp_REJ10/(num_samples-skip) );

		pwr_out << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
		pwr_out << bws(k) << " , "  << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
//		cout << "1%, 5%, and 10%-level rejection frequencies for Student's t-test" << endl;
//		cout << bws(k) << " , " << asymp_REJF1(k) << " , " << asymp_REJF5(k) << " , " << asymp_REJF10(k) << endl;
	}


	/* Calculate Student's t-test bootstrap p-values of the difference between two maximum squared Sharpe ratios */
	pwr_out << "k, h, boot_t_pvalue" << endl;
//	cout << "k, h, boot_t_pvalue" << endl;
	for (int k = 0; k < num_bws; ++k) {
		for (int h = 0; h < num_blocks; ++h) {
//			pwr_out << (boost::format("bw = %f and block size = %d") %bw %L).str() << endl;
			boot_REJ1 = 0; //reset rejection frequencies
			boot_REJ5 = 0;
			boot_REJ10 = 0;
			skip = 0;
			#pragma omp parallel for default(shared) reduction(+:skip,boot_REJ1,boot_REJ5,boot_REJ10) \
																schedule(static,CHUNK) private(i) firstprivate(boot_t_pvalue)
			for (i = 0; i < num_samples; ++i) {
				try {

					/* Calculate Student's t-test bootstrap p-value of the difference between two Sharpe ratios */
					boot_t_pvalue = TestSharpeR::t_boot_pvalue_diff_max_sq_sharpes_mimicking<kernel>(X[i], /* a T by N1 matrix of non-traded factors */
																									Y[i], /* a T by N2 matrix of non-traded factors */
																									Z_x[i], /* a T by M1 matrix of traded factors and basis asset returns */
																									Z_y[i], /* a T by M2 matrix of traded factors and basis asset returns */
																									bws(k), /* kernel bandwidth */
																									Ls(h), /* block size */
																									num_boots, /* number of bootstrap repetitions */
																									seed /* a seed to generate random numbers */);

					#pragma omp critical
					{
						pwr_out << bws(k) << " , " << Ls(h) << " , " << boot_t_pvalue <<  endl;
//						cout << bws(k) << " , " << Ls(h) << " , " << boot_t_pvalue <<  endl;
					}

					if (boot_t_pvalue <= 0.01)  ++boot_REJ1;//using 1%-level of significance
					if (boot_t_pvalue <= 0.05)  ++boot_REJ5;//using 5%-level of significance
					if (boot_t_pvalue <= 0.10)  ++boot_REJ10;//using 10%-level of significance

				}
				catch(...) { //catch all exceptions
					cerr << "Power::power_f: An exception occurs!" << endl;
					++skip;
				}
			}
			boot_REJF1(k,h) = ((double) boot_REJ1/(num_samples-skip) );
			boot_REJF5(k,h) = ((double) boot_REJ5/(num_samples-skip) );
			boot_REJF10(k,h) = ((double) boot_REJ10/(num_samples-skip) );

			pwr_out << "1%, 5%, and 10%-level rejection frequencies for the bootstrap Student's t-test" << endl;
			pwr_out << k << " , " << h << " , " << boot_REJF1(k,h) << " , " << boot_REJF5(k,h) << " , " << boot_REJF10(k,h) << endl;
//			cout << "1%, 5%, and 10%-level rejection frequencies for the bootstrap Student's t-test" << endl;
//			cout << k << " , " << h << " , " << boot_REJF1(k,h) << " , " << boot_REJF5(k,h) << " , " << boot_REJF10(k,h) << endl;
		}
	}

	/* Calculate the self-normalized test statistics of the difference between two maximum squared Sharpe ratios */
	pwr_out << "sn_stat" << endl;
	cout << "sn_stat" << endl;
	skip = 0;
	#pragma omp parallel for default(shared) reduction(+:skip,sn_REJ1,sn_REJ5,sn_REJ10) \
									schedule(static,CHUNK) private(i) firstprivate(sn_stat)
	for (i = 0; i < num_samples; ++i) {
		try {
			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn_max_sq_sharpe_stat_mimicking(X[i], /* a T by N1 matrix of non-traded factors */
																	Y[i], /* a T by N2 matrix of non-traded factors */
																	Z_x[i], /* a T by M1 matrix of traded factors and basis asset returns */
																	Z_y[i] /* a T by M2 matrix of traded factors and basis asset returns */);

			#pragma omp critical
			{
				pwr_out << sn_stat << endl;
	//			cout << sn_stat << endl;
			}

			//using the upper critical values tabulated in Table 1 of Lobato (2001)
			if (pow(sn_stat, 2.) >= 99.76) ++sn_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= 45.40) ++sn_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= 28.31) ++sn_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn_REJF1 = ((double) sn_REJ1/(num_samples-skip) );
	sn_REJF5 = ((double) sn_REJ5/(num_samples-skip) );
	sn_REJF10 = ((double) sn_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
//	cout << sn_REJF1 << " , " << sn_REJF5 << " , " << sn_REJF10 << endl;


	/* Calculate Volgushev and Shao's (2014) self-normalized test statistics of the difference between two maximum squared Sharpe ratios */
	pwr_out << "sn2_stat" << endl;
	cout << "sn2_stat" << endl;
	skip = 0;
	#pragma omp parallel for default(shared) reduction(+:skip,sn2_REJ1,sn2_REJ5,sn2_REJ10) \
									schedule(static,CHUNK) private(i) firstprivate(sn_stat)
	for (i = 0; i < num_samples; ++i) {
		try {
			/* Calculate the self-normalized t-statistic of the difference between two Sharpe ratios */
			sn_stat = SnTestSharpeR::sn2_max_sq_sharpe_stat_mimicking(X[i], /* a T by N1 matrix of non-traded factors */
																	Y[i], /* a T by N2 matrix of non-traded factors */
																	Z_x[i], /* a T by M1 matrix of traded factors and basis asset returns */
																	Z_y[i] /* a T by M2 matrix of traded factors and basis asset returns */);

			#pragma omp critical
			{
				pwr_out << sn_stat << endl;
	//			cout << sn_stat << endl;
			}

			//using the upper critical values tabulated in Table 2 of Shao (2015)
			if (pow(sn_stat, 2.) >= 139.73) ++sn2_REJ1;//using 1%-level of significance
			if (pow(sn_stat, 2.) >= 68.41) ++sn2_REJ5;//using 5%-level of significance
			if (pow(sn_stat, 2.) >= 44.46) ++sn2_REJ10;//using 10%-level of significance
		}
		catch(...) { //catch all exceptions
			cerr << "Power::power_f: An exception occurs while calculating sn_stat!" << endl;
			++skip;
		}
	}

	sn2_REJF1 = ((double) sn2_REJ1/(num_samples-skip) );
	sn2_REJF5 = ((double) sn2_REJ5/(num_samples-skip) );
	sn2_REJF10 = ((double) sn2_REJ10/(num_samples-skip) );

	pwr_out << "1%, 5%, and 10%-level rejection frequencies for the self-normalized t-test" << endl;
	pwr_out << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;
//	cout << "1%, 5%, and 10%-level rejection frequencies for Volgushev and Shao's (2014) self-normalized t-test" << endl;
//	cout << sn2_REJF1 << " , " << sn2_REJF5 << " , " << sn2_REJF10 << endl;


	return {asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10};
}
















#endif
