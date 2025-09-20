#ifndef TEST_SHARPE_H
#define TEST_SHARPE_H

#include <unistd.h>
#include <omp.h>
#include <cassert>
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

class TestSharpeR{
	public:
		TestSharpeR (){  }; //default constructor
		~TestSharpeR () { };//default destructor

		/* Calculate Student's t-statistic of the difference between two Sharpe ratios */
		template<double kernel(double) /* a HAC kernel */>
		static double t_stat_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
										  const matrix<double> &Y, /* excess returns on second asset */
										  const double bw /* kernel bandwidth */);

		/* Calculate Student's t-test p-value of the difference between two Sharpe ratios */
		template<double kernel(double) /* a HAC kernel */>
		static double t_pvalue_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
											const matrix<double> &Y, /* excess returns on second asset */
											const double bw /* kernel bandwidth */);


		/* Calculate Student's t-test bootstrap p-value of the difference between two Sharpe ratios */
		template<double kernel(double) /* a HAC kernel */>
		static double t_boot_pvalue_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
												const matrix<double> &Y, /* excess returns on second asset */
												const double bw, /* kernel bandwidth */
												const int B, /* block size */
												int num_boots, /* number of bootstrap repetitions */
												unsigned long seed = 12345 /* a seed to generate random numbers */);

		/* Calculate Student's t-statistic of the difference between two maximum squared Sharpe ratios */
		template<double kernel(double) /* a HAC kernel */>
		static double t_stat_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
												const matrix<double> &Y, /* a T by N2 matrix of factors */
												const double bw /* kernel bandwidth */);

		/* Calculate Student's t-test p-value of the difference between two maximum squared Sharpe ratios */
		template<double kernel(double) /* a HAC kernel */>
		static double t_pvalue_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
													const matrix<double> &Y, /* a T by N2 matrix of factors */
													const double bw /* kernel bandwidth */);

		/* Calculate Student's t-test bootstrap p-value of the difference between two maximum squared Sharpe ratios */
		template<double kernel(double) /* a HAC kernel */>
		static double t_boot_pvalue_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
														const matrix<double> &Y, /* a T by N2 matrix of factors */
														const double bw, /* kernel bandwidth */
														const int B, /* block size */
														const int num_boots, /* number of bootstrap repetitions */
														unsigned long seed = 123456/* a seed to generate random numbers */);

		/* Calculate Student's t-statistic of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
		[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
		template<double kernel(double) /* a HAC kernel */>
		static double t_stat_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
														   const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
														   const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
														   const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
														   const double bw /* kernel bandwidth */);

		/* Calculate Student's t-test p-value of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
		[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
		template<double kernel(double) /* a HAC kernel */>
		static double t_pvalue_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
														   const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
														   const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
														   const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
														   const double bw /* kernel bandwidth */);

		/* Calculate Student's t-test bootstrap p-value of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
		[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
		template<double kernel(double) /* a HAC kernel */>
		static double t_boot_pvalue_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
															    const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
															    const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
															    const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
																const double bw, /* kernel bandwidth */
																const int B, /* block size */
																const int num_boots, /* number of bootstrap repetitions */
																unsigned long seed = 123456 /* a seed to generate random numbers */);


};

/* Calculate Student's t-test p-value of the difference between two Sharpe ratios */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_pvalue_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
								const matrix<double> &Y, /* excess returns on second asset */
								const double bw /* kernel bandwidth */) {
	double t_stat = ( Sharpe::sharpe_ratio(X) - Sharpe::sharpe_ratio(Y) ) / Sharpe::se_diff_sharpes<kernel>(X, Y, bw);

	return 2*gsl_cdf_ugaussian_P(-fabs(t_stat));
}

/* Calculate Student's t-statistic of the difference between two Sharpe ratios */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_stat_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
										const matrix<double> &Y, /* excess returns on second asset */
										const double bw /* kernel bandwidth */) {
	double t_stat = ( Sharpe::sharpe_ratio(X) - Sharpe::sharpe_ratio(Y) ) / Sharpe::se_diff_sharpes<kernel>(X, Y, bw);

	return t_stat;
}

/* Calculate Student's t-test p-value of the difference between two maximum squared Sharpe ratios */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_pvalue_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
												 const matrix<double> &Y, /* a T by N2 matrix of factors */
												 const double bw /* kernel bandwidth */) {
	double t_stat = ( Sharpe::max_sq_sharpe_ratio(X) - Sharpe::max_sq_sharpe_ratio(Y) ) / Sharpe::se_diff_max_sq_sharpes<kernel>(X, Y, bw);

	return 2*gsl_cdf_ugaussian_P(-fabs(t_stat));
}

/* Calculate Student's t-statistic of the difference between two maximum squared Sharpe ratios */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_stat_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
												 const matrix<double> &Y, /* a T by N2 matrix of factors */
												 const double bw /* kernel bandwidth */) {
	double t_stat = ( Sharpe::max_sq_sharpe_ratio(X) - Sharpe::max_sq_sharpe_ratio(Y) ) / Sharpe::se_diff_max_sq_sharpes<kernel>(X, Y, bw);

	return t_stat;
}


/* Calculate Student's t-test p-value of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_pvalue_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
														   const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
													       const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
													       const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
												           const double bw /* kernel bandwidth */) {
	double t_stat = ( Sharpe::max_sq_sharpe_ratio_mimicking(X,Z_x) - Sharpe::max_sq_sharpe_ratio_mimicking(Y,Z_y) )
																	/ Sharpe::Sharpe::se_diff_max_sq_sharpes_mimicking<kernel>(X, Y, Z_x, Z_y, bw);

	return 2*gsl_cdf_ugaussian_P(-fabs(t_stat));
}

/* Calculate Student's t-statistic of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_stat_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
													   const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
													   const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
													   const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
													   const double bw /* kernel bandwidth */) {
	double t_stat = ( Sharpe::max_sq_sharpe_ratio_mimicking(X,Z_x) - Sharpe::max_sq_sharpe_ratio_mimicking(Y,Z_y) )
																	/ Sharpe::Sharpe::se_diff_max_sq_sharpes_mimicking<kernel>(X, Y, Z_x, Z_y, bw);

	return t_stat;
}

/* Calculate Student's t-test bootstrap p-value of the difference between two Sharpe ratios */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_boot_pvalue_diff_sharpes(const matrix<double> &X, /* excess returns on first asset */
								const matrix<double> &Y, /* excess returns on second asset */
								const double bw, /* kernel bandwidth */
								const int B, /* block size */
								const int num_boots, /* number of bootstrap repetitions */
								unsigned long seed /* a seed to generate random numbers */) {
	/* Calculate the sample t-statistic */
	double t_stat = ( Sharpe::sharpe_ratio(X) - Sharpe::sharpe_ratio(Y) ) / Sharpe::se_diff_sharpes<kernel>(X, Y, bw);

	/* Calculate the bootstrap t-statistics */
	int T = X.nr();
	matrix<double> Z(T,2);
	Z = join_rows(X,Y);

	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

	matrix<double> Z_bt(T,2);
	double boot_t_stat = 0., boot_se_diff_sharpe = 0.;
	int count1 = 0, max_iter = num_boots;
	while (max_iter-- > 0) {
//		cout << "max_iter = " << max_iter << endl;
		//generate bootstrapped samples
		rseed = gsl_rng_get(r); //generate a random seed
		Z_bt = Bootstrap::CBB(Z, /* a T by N matrix */
							  B, /* block size */
							  rseed /* a seed to generate random numbers */);

		boot_se_diff_sharpe = Sharpe::boot_se_diff_sharpes(colm(Z_bt,0), colm(Z_bt,1), B);
//		cout << "boot se boot_se_diff_sharpe" << boot_se_diff_sharpe << endl;
		boot_t_stat = ( Sharpe::sharpe_ratio(colm(Z_bt,0)) - Sharpe::sharpe_ratio(colm(Z_bt,1)) - ( Sharpe::sharpe_ratio(X) - Sharpe::sharpe_ratio(Y) ) )
																																				/ boot_se_diff_sharpe;
//		cout << "boot_t_stat = " << boot_t_stat << " , " << t_stat << endl;
		if (fabs(boot_t_stat) >= fabs(t_stat)) {
			count1++;
		}
	}

	double boot_pvalue = (count1+1.)/(num_boots+1);


	gsl_rng_free(r); //free memory

	return boot_pvalue;
}


/* Calculate Student's t-test bootstrap p-value of the difference between two maximum squared Sharpe ratios */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_boot_pvalue_diff_max_sq_sharpes(const matrix<double> &X, /* a T by N1 matrix of factors */
														const matrix<double> &Y, /* a T by N2 matrix of factors */
														const double bw, /* kernel bandwidth */
														const int B, /* block size */
														const int num_boots, /* number of bootstrap repetitions */
														unsigned long seed /* a seed to generate random numbers */) {
	/* Calculate the sample t-statistic */
	double t_stat = ( Sharpe::max_sq_sharpe_ratio(X) - Sharpe::max_sq_sharpe_ratio(Y) ) / Sharpe::se_diff_max_sq_sharpes<kernel>(X, Y, bw);

	/* Calculate the bootstrap t-statistics */
	ASSERT_(X.nr() == Y.nr());
	int T = X.nr(), N1 = X.nc(), N2 = Y.nc();
	matrix<double> Z(T,N1+N2);
	Z = join_rows(X,Y);

	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

	matrix<double> Z_bt(T,N1+N2);
	double boot_t_stat = 0., boot_se_diff_sharpe = 0.;
	int count1 = 0, max_iter = num_boots;
	while (max_iter-- > 0) {
//		cout << "max_iter = " << max_iter << endl;
		//generate bootstrapped samples
		rseed = gsl_rng_get(r); //generate a random seed
		Z_bt = Bootstrap::CBB(Z, /* a T by N matrix */
							  B, /* block size */
							  rseed /* a seed to generate random numbers */);

		boot_se_diff_sharpe = Sharpe::boot_se_diff_max_sq_sharpes(colm(Z_bt, range(0,N1-1)), /* a T by N1 matrix of bootstrapped factors */
																  colm(Z_bt, range(N1,N1+N2-1)), /* a T by N2 matrix of bootstrapped factors */
																  B /* block size */);

//		cout << "boot se boot_se_diff_max_sq_sharpe" << boot_se_diff_sharpe << endl;
		boot_t_stat = ( Sharpe::max_sq_sharpe_ratio(colm(Z_bt, range(0,N1-1))) - Sharpe::max_sq_sharpe_ratio(colm(Z_bt, range(N1,N1+N2-1))) \
										- ( Sharpe::max_sq_sharpe_ratio(X) - Sharpe::max_sq_sharpe_ratio(Y) ) ) / boot_se_diff_sharpe;
//		cout << "boot_t_stat = " << boot_t_stat << " , " << t_stat << endl;
		if (fabs(boot_t_stat) >= fabs(t_stat)) {
			count1++;
		}
	}

	double boot_pvalue = (count1+1.)/(num_boots+1);


	gsl_rng_free(r); //free memory

	return boot_pvalue;
}


/* Calculate Student's t-test bootstrap p-value of the difference between two maximum squared Sharpe ratios attainable from two sets of non-traded factors
[see Proposition 4 of Barillas, Kan, Robotti & Shanken (2020)] */
template<double kernel(double) /* a HAC kernel */>
double TestSharpeR::t_boot_pvalue_diff_max_sq_sharpes_mimicking(const matrix<double> &X, /* a T by N1 matrix of non-traded factors */
															    const matrix<double> &Y, /* a T by N2 matrix of non-traded factors */
															    const matrix<double> &Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
															    const matrix<double> &Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
																const double bw, /* kernel bandwidth */
																const int B, /* block size */
																const int num_boots, /* number of bootstrap repetitions */
																unsigned long seed /* a seed to generate random numbers */) {
	/* Calculate the sample t-statistic */
	double t_stat = TestSharpeR::t_stat_diff_max_sq_sharpes_mimicking<kernel>(X, /* a T by N1 matrix of non-traded factors */
																				Y, /* a T by N2 matrix of non-traded factors */
																				Z_x, /* a T by M1 matrix of traded factors and basis asset returns */
																				Z_y, /* a T by M2 matrix of traded factors and basis asset returns */
																				bw /* kernel bandwidth */);

	/* Calculate the bootstrap t-statistics */
	assert(X.nr() == Y.nr() && Z_x.nr() == Z_y.nr() && X.nr() == Z_x.nr());
	int T = X.nr(), N1 = X.nc(), N2 = Y.nc(), M1 = Z_x.nc(), M2 = Z_y.nc();
	matrix<double> Z(T,N1+N2+M1+M2);
	Z = join_rows(join_rows(X,Z_x), join_rows(Y,Z_y));
//	cout << rowm(Z,10) << endl;

	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

	matrix<double> Z_bt(T,N1+N2+M1+M2);
	matrix<double> X_bt(T,N1), Z_x_bt(T,M1), Y_bt(T,N2), Z_y_bt(T,M2);


	double boot_t_stat = 0., boot_se_diff_sharpe = 0.;
	int count1 = 0, max_iter = num_boots;
	while (max_iter-- > 0) {
//		cout << "max_iter = " << max_iter << endl;
		//generate bootstrapped samples
		rseed = gsl_rng_get(r); //generate a random seed
		Z_bt = Bootstrap::CBB(Z, /* a T by N matrix */
							  B, /* block size */
							  rseed /* a seed to generate random numbers */);
//		cout << rowm(Z_bt, 20) << endl;

		X_bt = colm(Z_bt, range(0,N1-1));
		Z_x_bt = colm(Z_bt, range(N1,N1+M1-1));
		Y_bt = colm(Z_bt, range(N1+M1,N1+M1+N2-1));
		Z_y_bt = colm(Z_bt, range(N1+M1+N2,N1+M1+N2+M2-1));


		boot_se_diff_sharpe = Sharpe::boot_se_diff_max_sq_sharpes_mimicking(X_bt, /* a T by N1 matrix of bootstrapped non-traded factors */
																			Y_bt, /* a T by N2 matrix of bootstrapped non-traded factors */
																			Z_x_bt, /* a T by M1 matrix of bootstrapped traded factors and basis asset returns */
																			Z_y_bt, /* a T by M2 matrix of bootstrapped traded factors and basis asset returns */
																			B /* block size */);

//		cout << "boot se boot_se_diff_max_sq_sharpe" << boot_se_diff_sharpe << endl;
		boot_t_stat = ( Sharpe::max_sq_sharpe_ratio_mimicking(X_bt, Z_x_bt) - Sharpe::max_sq_sharpe_ratio_mimicking(Y_bt, Z_y_bt)  \
										- ( Sharpe::max_sq_sharpe_ratio_mimicking(X, Z_x) - Sharpe::max_sq_sharpe_ratio_mimicking(Y, Z_y) ) ) / boot_se_diff_sharpe;
//		cout << "boot_t_stat = " << boot_t_stat << " , " << t_stat << endl;

		if (fabs(boot_t_stat) >= fabs(t_stat)) {
			count1++;
		}
	}

	double boot_pvalue = (count1+1.)/(num_boots+1);


	gsl_rng_free(r); //free memory

	return boot_pvalue;
}











#endif
