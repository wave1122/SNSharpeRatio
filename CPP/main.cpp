#include <iostream>
#include <fstream>
#include <iomanip>   // format manipulation
#include <string>
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <numeric>
#include <stdio.h>
#include <map>
#include <unistd.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_bspline.h>
#include <vector> // C++ vector class
#include <algorithm>
#include <functional>
#include <gsl/gsl_randist.h>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <gsl/gsl_rng.h>
#include <unistd.h>
#include <filein.h>
#include <limits>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <chrono>
//#include <windows.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>

//#include <shogun/mathematics/Math.h>
//#include <shogun/mathematics/Statistics.h>
//#include <shogun/lib/SGSparseVector.h>
//#include <shogun/lib/config.h>
//#include <shogun/base/init.h>
//#include <shogun/base/some.h>
//#include <shogun/ensemble/MajorityVote.h>
//#include <shogun/evaluation/MeanSquaredError.h>
//#include <shogun/labels/RegressionLabels.h>
//#include <shogun/lib/SGMatrix.h>
//#include <shogun/lib/SGVector.h>
//#include <shogun/lib/SGString.h>
//#include <shogun/loss/SquaredLoss.h>
//#include <shogun/machine/RandomForest.h>
//#include <shogun/machine/StochasticGBMachine.h>
//#include <shogun/multiclass/tree/CARTree.h>
//#include <shogun/util/iterators.h>
//#include <shogun/mathematics/linalg/LinalgNamespace.h>
//#include <shogun/mathematics/linalg/linop/MatrixOperator.h>
//
//#include <shogun/labels/BinaryLabels.h>
//#include <shogun/features/DenseFeatures.h>
//#include <shogun/kernel/GaussianKernel.h>
//#include <shogun/classifier/svm/LibSVM.h>
//#include <shogun/lib/common.h>
//#include <shogun/io/SGIO.h>
//#include <shogun/io/File.h>

//#include <ShogunML/data/data.h>
//#include <matrix_ops2.h>
//#include <kernel.h>
//#include <power.h>
//#include <tests.h>


#include <plot.h>
#include <utils.h>


#define CHUNK 1

#include <kernel.h>
#include <tests3.h>
#include <dgp.h>
#include <sharpe_ratios.h>
//#include <bootstraps.h>
#include <test_sharpe_ratios.h>
#include <sn_test_sharpe_ratios.h>
#include <power.h>



using namespace std;
//namespace fs = std::experimental::filesystem;
//using namespace shogun;
//using namespace shogun::linalg;
using namespace dlib;

//void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

int main(void) {
	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;



//	int T = 15, N = 2;
//	matrix<double> X(T,N), Y(T,N);
//	X = randm(T,N);
//	Y = randm(T,N);
//
//	auto mean = Sharpe::mean(Y);
//	cout << mean << endl;
//
//	auto cov = Sharpe::covariance(X);
//	cout << cov << endl;
//
//	double sr = Sharpe::max_sq_sharpe_ratio(X);
//	cout << sr << endl;
//
//	double bw = 4.;
//	double se = Sharpe::se_diff_max_sq_sharpes<QS_kernel>(X,Y,bw);
//	cout << "se = " << se << endl;

//
//	auto Y = Bootstrap::CBB(X, L);
//	cout << Y << endl;
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
//	test_output << X << endl;
//	test_output << Y << endl;
//	test_output.close();


//    double bw = 10;
//    int T = 100;
//    int L = 30;
//    matrix<double> X(T,1), Y(T,1);
//    X = randm(T,1);
//    Y = randm(T,1);


	matrix<double> bws(5,1); /* kernel bandwidths */
	bws = 3, 6, 9, 12, 14;
	matrix<int> Ls(5,1); /* block sizes */
	Ls = 5, 10, 30, 50, 80;
	const int num_samples = 500; /* number of random samples */
	const int T = 500; /* sample size */
	const double nu = 6; /* parameter of the error distribution */
//	matrix<double> mu(2,1); /* means */
//	mu = 16.5/52,
//		 16.5/52;
//	matrix<double> A(2,2), B(2,2), C(2,2); /* GARCH coefficients */
//	A = 0.075, 0.050,
//		0.050, 0.075;
//	B = 0.90, 0.89,
//		0.89, 0.90;
//	C = 0.15, 0.13,
//		0.13, 0.15;
//	const string err_dist = "Student";
//	unsigned long seed = 123456;

//	auto X = Dgp::gen_dvech_mgarch<gsl_ran_tdist>(num_samples, /* number of random samples */
//														T, /* sample size */
//														nu, /* parameter of the error distribution */
//														mu, /* means */
//														A, /* ARCH coefficients */
//														B, /* AR coefficients */
//														C, /* intercepts */
//														err_dist, /* distribution of the errors */
//														seed /* a seed to generate random numbers */);
//	for (int i = 0; i < num_samples; ++i)
//		cout << "Sample = " << i << ":\n" << X[i] << endl;

//	const int num_samples = 100; /* number of random samples */
//	const int T = 500; /* sample size */
//	const double nu = 6; /* parameter of the error distribution */
	matrix<double> mu1(1,1);
	mu1(0) = 0.5;
	matrix<double> mu(4,1);
	mu = 0.5,
		 0.5,
		 0.5,
		 0.5;
	matrix<double> A(4,4), B(4,4), Omega(4,4), Omega1(3,3); /* GARCH coefficients */
	A = 0.075, 0.050, 0.010, 0.100,
		0.050, 0.025, 0.020, 0.050,
		0.010, 0.020, 0.100, 0.090,
		0.100, 0.050, 0.090, 0.005;
	B = 0.90, 0.89, 0.10, 0.50,
		0.89, 0.50, 0.80, 0.10,
		0.10, 0.80, 0.15, 0.20,
		0.50, 0.10, 0.20, 0.05;
	Omega = 0.20, 0.05, -0.01, 0.05,
			0.05, 0.50, -0.08, 0.02,
		   -0.01,-0.08,  0.90, 0.04,
		    0.05, 0.02,  0.04, 1.00;
//	Omega = 0.20, 0.05, 0.05, 0.05,
//			0.05, 0.20, 0.05, 0.05,
//		    0.05, 0.05, 0.20, 0.05,
//		    0.05, 0.05, 0.05, 0.20;
	Omega1 = subm(Omega, range(0,2), range(0,2));
	const string err_dist = "Student";
	unsigned long seed = 123456;
	cout << A << endl;
	cout << B << endl;
	cout << Omega << endl;

//	auto X = Dgp::gen_dvech_trigarch<gsl_ran_tdist>(num_samples, /* number of random samples */
//													T, /* sample size */
//													nu, /* parameter of the error distribution */
//													mu1, /* mean of the overlapping factor */
//													A, /* ARCH coefficients */
//													B, /* AR coefficients */
//													Omega, /* unconditional covariance matrix */
//													err_dist, /* distribution of the errors */
//													seed /* a seed to generate random numbers */);
//	for (int i = 0; i < num_samples; ++i) {
//		cout << "Sample = " << i << ":\n" << X[i] << endl;
//		cout << "Sample = " << i << ":\n" << endl;
//		cout << colm(X[i], range(0,1)) << endl;
//		cout << colm(X[i], range(0,2,2)) << endl;
//	}

//	double pvalue = 0.;
//	double bw = 10.;
//	int L = 20;
//	int num_boots = 499;
//	for (const auto &Xi: X) {
////		se = Sharpe::boot_se_diff_max_sq_sharpes(colm(Xi,range(0,1)), /* a T by N1 matrix of bootstrapped factors */
////												colm(Xi,range(0,2,2)), /* a T by N2 matrix of bootstrapped factors */
////												L /* block size */);
//		pvalue = TestSharpeR::t_boot_pvalue_diff_max_sq_sharpes<QS_kernel>(colm(Xi,range(0,1)), /* a T by N1 matrix of factors */
//																			colm(Xi,range(0,2,2)), /* a T by N2 matrix of factors */
//																			bw, /* kernel bandwidth */
//																			L, /* block size */
//																			num_boots, /* number of bootstrap repetitions */
//																			seed /* a seed to generate random numbers */);
//		cout << pvalue << endl;
//	}

//	/* Calculate the self-normalized test statistic for the difference between two maximum squared Sharpe ratios */
//	double sn_tstat = 0.;
//	for (const auto &Xi: X) {
//		sn_tstat = SnTestSharpeR::sn_max_sq_sharpe_stat(colm(Xi, range(0,1)), /* a T by N1 matrix of factors */
//														colm(Xi, range(0,2,2)) /* a T by N2 matrix of factors */);
////		cout << Xi << endl;
//		cout << sn_tstat << endl;
////		break;
//	}


//
//	double bw = 4.;
//	for (const auto &Xi: X) {
//		double se = Sharpe::se_diff_max_sq_sharpes<QS_kernel>(colm(Xi, range(0,1)), colm(Xi, range(0,2,2)), bw);
//		cout << "se = " << se << endl;
//	}
//
//	double sr1, sr2;
//	for (const auto &Xi: X) {
//		sr1 = Sharpe::max_sq_sharpe_ratio(colm(Xi, range(0,1)));
//		sr2 = Sharpe::max_sq_sharpe_ratio(colm(Xi, range(0,2,2)));
//		cout << "sr1 = " << sr1 << " sr2 = " << sr2 << endl;
//	}

//	double sn_tstat = 0.;
//	for (const auto &Xi: X) {
//		sn_tstat = SnTestSharpeR::sn_sharpe_stat(colm(Xi,0), /* excess returns on first asset */
//													colm(Xi,1) /* excess returns on second asset */);
//		cout << sn_tstat << endl;
//		break;
//	}

	const int num_boots = 499;

//	unsigned long seed = 12345; /* a seed to generate random numbers */

	ofstream pwr_out;
	pwr_out.open("./output.txt", ios::out);
	const bool fovl = true; /* True if there are overlapping factors */

//	auto [asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10]
//											= Power::power_f<QS_kernel,
//															Dgp::gen_dvech_garch<gsl_ran_tdist>>(num_samples, /* number of random samples */
//																								T, /* sample size */
//																								bws, /* kernel bandwidths */
//																								Ls, /* block sizes */
//																								num_boots, /* number of bootstrap repetitions */
//																								nu, /* parameter of the error distribution */
//																								mu, /* means */
//																								A, /* ARCH coefficients */
//																								B, /* AR coefficients */
//																								C, /* intercepts */
//																								err_dist, /* distribution of the errors */
//																								pwr_out, /* output stream */
//																								seed /* a seed to generate random numbers */);

	auto [asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10]
										 = Power::power_fovl<QS_kernel,
														Dgp::gen_dvech_trigarch<gsl_ran_tdist>>(num_samples, /* number of random samples */
																								T, /* sample size */
																								bws, /* kernel bandwidths */
																								Ls, /* block sizes */
																								num_boots, /* number of bootstrap repetitions */
																								nu, /* parameter of the error distribution */
																								mu1, /* means */
																								A, /* ARCH coefficients */
																								B, /* AR coefficients */
																								Omega1, /* unconditional covariance matrix */
																								err_dist, /* distribution of the errors */
																								true, /* True if there are overlapping factors */
																								pwr_out, /* output stream */
																								seed /* a seed to generate random numbers */);

//	auto [asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10]
//										 = Power::power_fovl<QS_kernel,
//														Dgp::gen_dvech_mgarch<gsl_ran_tdist>>(num_samples, /* number of random samples */
//																								T, /* sample size */
//																								bws, /* kernel bandwidths */
//																								Ls, /* block sizes */
//																								num_boots, /* number of bootstrap repetitions */
//																								nu, /* parameter of the error distribution */
//																								mu, /* means */
//																								A, /* ARCH coefficients */
//																								B, /* AR coefficients */
//																								Omega, /* unconditional covariance matrix */
//																								err_dist, /* distribution of the errors */
//																								fovl, /* True if there are overlapping factors */
//																								pwr_out, /* output stream */
//																								seed /* a seed to generate random numbers */);

	pwr_out.close();

//    auto [X, Y] = Dgp::gen_dvech_garch<gsl_ran_tdist>(num_samples, /* number of random samples */
//														T, /* sample size */
//														nu, /* parameter of the error distribution */
//														mu, /* means */
//														A, /* ARCH coefficients */
//														B, /* AR coefficients */
//														C, /* intercepts */
//														err_dist, /* distribution of the errors */
//														seed /* a seed to generate random numbers */);
//	cout << X.nr() << " , " << X.nc() << endl;
//	cout << "Sample of Y:" << endl;
//	cout << Y <<endl;
//    cout << X << endl;
//	double se = Sharpe::se_diff_sharpes<QS_kernel>(X, /* excess returns on first asset */
//														 Y, /* excess returns on second asset */
//														 bw /* kernel bandwidth */);
//	cout << se << endl;
//	double se = Sharpe::boot_se_diff_sharpes(X, /* bootstrapped excess returns on first asset */
//											Y, /* bootstrapped excess returns on second asset */
//											B /* block size */);
//	cout << se << endl;

//	auto pvalue = TestSharpeR::t_pvalue_diff_sharpes<QS_kernel>(X, /* excess returns on first asset */
//																 Y, /* excess returns on second asset */
//																 bw /* kernel bandwidth */);
//	cout << pvalue << endl;

//	int num_boots = 999;
////	unsigned long seed = 123456;
//	auto pvalue = TestSharpeR::t_boot_pvalue_diff_sharpes<QS_kernel>(X, /* excess returns on first asset */
//																	 Y, /* excess returns on second asset */
//																	 bw, /* kernel bandwidth */
//																	 L, /* block size */
//																	 num_boots, /* number of bootstrap repetitions */
//																	 seed /* a seed to generate random numbers */);
//	cout << pvalue << endl;


//
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
//	test_output << X << endl;
//	test_output << "Sample Y:" << endl;
//	test_output << Y << endl;
//	test_output.close();

//	gsl_rng * r;
//    const gsl_rng_type * gen;//random number generator
//    gsl_rng_env_setup();
//    gen = gsl_rng_taus;
//    r = gsl_rng_alloc(gen);
//	unsigned long seed = 12345; //113424;
//    gsl_rng_set(r, seed);

//    matrix<double> M(3,2);
//	M = 54.2,  7.4,
//        1,     2,
//        5.9,   0.05;
//	cout << M << endl;
//	double mean_M = dlib::mean(colm(M,1));
//	cout << mean_M << endl;

//    // Generate a Gaussian multiplier sequence
//    int num_samples = 5, T = 10, bandwidth = 5;
//	auto X = Dgp::gen_mgaussian<bartlett_kernel>(num_samples, /* number of random samples */
//													T, /* sample size */
//														bandwidth, /* bandwidth */
//														seed /* a seed to generate random numbers */);
//	cout << X << endl;
//	cout << X.nr() << endl;

//    // Calculate the asymptotic critical values
//	const double asymp_CV5 = std::pow(gsl_cdf_ugaussian_Pinv(1- (0.05/2.0)), 2.0);
//	const double asymp_CV10 = std::pow(gsl_cdf_ugaussian_Pinv(1- (0.10/2.0)), 2.0);
//	matrix<double> asymp_CV(2,1);
//	asymp_CV = asymp_CV5, asymp_CV10;
//
//	// Set the VAR(2) coefficients
//	matrix<double> A(2,2), B(2,2);
//    A = -0.1, 0.5,
//		 0.0, 0.0;
//	B = 0.1, -0.1,
//		0.2, -0.5;
//
//	matrix<double> A1(2,2), B1(2,2);
//	A1 = -0.1, 0.5,
//		 0.5, -0.5;
//	B1 = 0.1, -0.1,
//		0.2, -0.5;
//
//	const int num_samples = 500;
//	const int num_boots = 499;
//	int T0 = 0, T1 = 0, T = 300;
//	constexpr size_t num_trees = 500;
//	const double feature_subsampling_fraction = 1.0; /* fraction of features to consider at each node */
//	const int min_samples_per_leaf = 10; /* minimum samples per leaf node */
//	bool boot_lrvar = false; /* bootstrap the long-run variance if True */
//	const double kappa = 1.0;
//	const int bandwidth = 10;
//
//	ofstream size_out, pwr_out;
//
//	while (T <= 600) {
//		T0 = std::nearbyint((double) T/1.5);
//		T1 = T-T0-2;
//		cout << (boost::format("T=%d, T0=%d, T1=%d, bandwidth=%d") %T %T0 %T1 %bandwidth).str() << endl;
//
//		/* Calculate sizes */
//		string size_filename = "size_T_" + std::to_string(T) + "_T0_" + std::to_string(T0) + "_bandwidth_" + std::to_string(bandwidth) + "_use_lrvar_" \
//								+ std::to_string(boot_lrvar) + ".txt";
//		size_out.open(size_filename, ios::out);
//
//		auto [asymp_REJF, bootstrap_REJF, empir_CV] = Power::cValue<Dgp::VAR2, Dgp::gen_mgaussian<bartlett_kernel>> (asymp_CV5, /* 5% asymptotic CV */
//																													asymp_CV10, /* 10% asymptotic CV */
//																													A, /* AR(1) coefficients */
//																													B, /* AR(2) coefficients */
//																													T, /* sample size */
//																													T0, /* training sample size */
//																													T1, /* testing sample size */
//																													num_samples, /* number of random samples */
//																													num_boots, /* number of bootstraps */
//																													num_trees, /* number of trees */
//																													feature_subsampling_fraction, /* fraction of features to consider at each node */
//																													min_samples_per_leaf, /* minimum samples per leaf node */
//																													size_out, /* output stream */
//																													boot_lrvar, /* bootstrap the long-run variance if True */
//																													kappa, /* a distance exponent */
//																													bandwidth, /* Bartlett kernel bandwidth */
//																													seed /* a seed to generate random numbers */);
//		cout << "asymptotic rejection rates: " << asymp_REJF << endl;
//		cout << "bootstrap rejection rates: " << bootstrap_REJF << endl;
//		cout << "empirical CVs: " << empir_CV << endl;
//		size_out.close();
//
//		/* Calculate powers */
//		string pwr_filename = "power_T_" + std::to_string(T) + "_T0_" + std::to_string(T0) + "_bandwidth_" + std::to_string(bandwidth) + "_use_lrvar_" \
//								+ std::to_string(boot_lrvar) + ".txt";
//		pwr_out.open(pwr_filename, ios::out);
//		auto [empir_REJF1, asymp_REJF1, bootstrap_REJF1] = Power::power_f<Dgp::VAR2, Dgp::gen_mgaussian<bartlett_kernel>>(empir_CV, /*5% and 10% empirical critical values*/
//																														 asymp_CV, /*5% and 10% asymptotic critical values*/
//																														 A1, /* AR(1) coefficients */
//																														 B1, /* AR(2) coefficients */
//																														 T, /* sample size */
//																														 T0, /* training sample size */
//																														 T1, /* testing sample size */
//																														 num_samples, /* number of random samples */
//																														 num_boots, /* number of bootstraps */
//																														 num_trees, /* number of trees */
//																														 feature_subsampling_fraction, /* fraction of features to consider at each node */
//																														 min_samples_per_leaf, /* minimum samples per leaf node */
//																														 pwr_out, /* output stream */
//																														 boot_lrvar, /* bootstrap the long-run variance if True */
//																														 kappa, /* a distance exponent */
//																														 bandwidth, /* Bartlett kernel bandwidth */
//																														 seed /* a seed to generate random numbers */);
//		cout << "empirical rejection rates: " << empir_REJF1 << endl;
//		cout << "asymptotic rejection rates: " << asymp_REJF1 << endl;
//		cout << "bootstrap rejection rates: " << bootstrap_REJF1 << endl;
//		pwr_out.close();
//
//		T += 100;
//	}


//    matrix<double> A(2,2), B(2,2);
//    A = -0.1, 0.5,
//		 0.5, -0.5;
//	B = 0.1, -0.1,
//		0.2, -0.5;
//
//	int T = 300, T0 = std::nearbyint((double) T/1.5), T1 = T-T0;
//	cout << (boost::format("T=%d, T0=%d, T1=%d") %T %T0 %T1).str() << endl;
//
//    auto [sample_X, sample_Y] = Dgp::VAR2(A, /* AR(1) coefficients */
//											B, /* AR(2) coefficients */
//											T, /* sample size */
//											seed);
//	cout << sample_X << endl;
//	cout << "---------------------" << endl;
//	cout << sample_Y << endl;

//	matrix<double> X = join_rows(rowm(sample_X, range(0,T-3)), rowm(sample_X, range(1,T-2)));
//	matrix<double> Y = rowm(sample_Y, range(2,T-1));
//	matrix<double> Z = join_rows(rowm(sample_Y, range(0,T-3)), rowm(sample_Y, range(1,T-2)));
//	cout << Y << endl;
//	cout << Y.nr() << endl;
//	cout << Y.nc() << endl;
//	cout << Z << endl;
//	cout << Z.nr() << endl;
//
//	matrix<double> train_X = rowm(X, range(0,T0-1)), train_Y = rowm(Y, range(0,T0-1)), train_Z = rowm(Z, range(0,T0-1));
//	cout << "training samples:" << endl;
//	cout << train_X << endl;
//	cout << train_Y << endl;
//	cout << train_Z << endl;
//	cout << train_Z.nr() << endl;
//	matrix<double> test_X = rowm(X, range(T0,X.nr()-1)), test_Y = rowm(Y, range(T0,Y.nr()-1)), test_Z = rowm(Z, range(T0,Z.nr()-1));
//	cout << "test samples:" << endl;
//	cout << test_X.nr() << endl;
//	cout << test_Y.nr() << endl;
//	cout << test_Z << endl;
//	cout << test_Z.nr() << endl;
//	T1 = test_X.nr();
//
//	constexpr size_t num_trees = 500;
//	const double feature_subsampling_fraction = 1.0; /* fraction of features to consider at each node */
//	const int min_samples_per_leaf = 10; /* minimum samples per leaf node */
//	bool boot_lrvar = false;
//	double kappa = 1.0;
//	int bandwidth = 10;
//
//	// Generate a Gaussian multiplier sequence
//    int num_boots = 499;
//	auto xi = Dgp::gen_mgaussian<bartlett_kernel>(num_boots, /* number of random samples */
//													T1, /* sample size */
//														bandwidth, /* bandwidth */
//														seed /* a seed to generate random numbers */);
//	cout << xi.nc() << endl;
//	cout << xi.nr() << endl;
//
//	// Test the GC bootstrap statistic
//	auto [boot_pvalue, gc_stat, boot_stats] = GC_TEST::gc_stat_bt( train_X, /* a T0 by N matrix of training observations on covariates */
//																	train_Y, /* a T0 by M matrix of training observations on the targets */
//																	train_Z, /* a T0 by K matrix of training observations on controls */
//																	test_X, /* a T1 by N matrix of test observations on covariates and controls */
//																	test_Y, /* a T1 by M matrix of test observations on the targets */
//																	test_Z, /* a T1 by K matrix of test observations on controls */
//																	xi, /* a T1 by num_boots matrix of dependent multipliers */
//																	num_trees, /* number of trees */
//																	feature_subsampling_fraction, /* fraction of features to consider at each node */
//																	min_samples_per_leaf, /* minimum samples per leaf node */
//																	boot_lrvar, /* bootstrap the long-run variance if True */
//																	kappa, /* a distance exponent */
//																	bandwidth,  /* Bartlett kernel bandwidth */
//																	seed  /* a seed */);
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
////	test_output << "xi = \n" << xi << endl;
//	test_output << "Bootstrap Granger distance p-value = " << boot_pvalue << endl;
//	test_output << "Granger distance statistic = " << gc_stat << endl;
//	test_output << "Bootstrap Granger distance statistics = \n" << boot_stats << endl;
//	test_output.close();



	// Test the GC statistic
//	double gc_stat = GC_TEST::gc_stat(	train_X, /* a T0 by N matrix of training observations on covariates */
//										train_Y, /* a T0 by M matrix of training observations on the targets */
//										train_Z, /* a T0 by K matrix of training observations on controls */
//										test_X, /* a T1 by N matrix of test observations on covariates and controls */
//										test_Y, /* a T1 by M matrix of test observations on the targets */
//										test_Z, /* a T1 by K matrix of test observations on controls */
//										num_trees, /* number of trees */
//										feature_subsampling_fraction, /* fraction of features to consider at each node */
//										min_samples_per_leaf, /* minimum samples per leaf node */
//										kappa, /* a distance exponent */
//										bandwidth  /* Bartlett kernel bandwidth */);
//
//	double level_sig = 0.05; // set a nominal size
//	double CV = std::pow(gsl_cdf_ugaussian_Pinv(1- (level_sig/2.0)), 2.0);
//	cout << gc_stat << endl;
//	cout << CV << endl;
//
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
//	test_output << "Granger distance statistic = " << gc_stat << endl;
//	test_output << "CV = " << CV << endl;
//	test_output.close();

//	matrix<double> M(3,3), N(1,3);
//	M = 1, -1, 3,
//		2, -2, -0.5,
//		0.5, -0.8, 1;
//	N = 1, 2, -3;
////	cout << M << endl;
//	cout << N << endl;
//	cout << N(0,0) << " , " << N(1) << " , " << N(2) << endl;
//
//	matrix<double> A(3,1);
//	A = rowm(M,0) - N;
//	cout << A << endl;
//	Samples Z1(3);
//	for (auto i = 0; i < 3; ++i) {
//		Z1[i] = rowm(M, i);
//		cout << (boost::format("i=%d, Z1[i]=%f") % i % Z1[i]).str() << endl;
//	}
//	cout << "pow(3, 2)=" << pow(3,2.) << endl;
//
//	cout << std::norm(x) << endl;

//	constexpr DataType start = -10;
//	constexpr DataType end = 10;
//	constexpr size_t num_samples = 1000;
//	constexpr size_t num_trees = 1000;
//	const double feature_subsampling_fraction = 0.5; /* fraction of features to consider at each node */
//	const int min_samples_per_leaf = 100; /* minimum samples per leaf node */


//	int T0 = 1000, T1 = 200, N = 20, M = 20, K = 5;

//	matrix<double> test_X(1,N), train_Y1(T0,1);
//	Samples train_X(T0);
//	for (int i = 0; i < T0; ++i)
//		train_X[i] = randm(N, 1);
//	test_X = randm(T1,N);
//
//	std::vector<double> train_Y(T0);
//	train_Y1 = randm(T0,1);
//	for (int i = 0; i < T0; ++i) {
//		train_Y[i] = train_Y1(i);
//	}

//	// test Random Forest
//	matrix<double> est = ML_REG_DCORR::RF1(train_X, /* a training sample */
//										train_Y, /* training labels */
//										test_X, /* a test sample */
//										num_trees, /* number of trees */
//										feature_subsampling_fraction, /* fraction of features to consider at each node */
//										min_samples_per_leaf);
//	cout << est << endl;

//	// test the conditional expectations of distances
//	matrix<double> train_Xt = randm(T0,N), train_Zt = randm(T0,M), Xs = randm(1,N), Zt = randm(T1,M);
//	double kappa = 1.5;
//	matrix<double> est = ML_REG_DCORR::E_ts(train_Xt, /* a T0 by N matrix */
//									 train_Zt, /* a T0 by M matrix */
//									 Xs, /* a 1 by N row vector */
//									 Zt, /* a 1 by M row vector */
//									 num_trees, /* number of trees */
//									 feature_subsampling_fraction, /* fraction of features to consider at each node */
//									 min_samples_per_leaf, /* minimum samples per leaf node */
//									 kappa /* a distance exponent */);
//
//	cout << "estimate = " << est << endl;

//	// test the double conditional expectations of distances
//	matrix<double> train_X = randm(T0,N), train_Z = randm(T0,M), test_X = randm(T1,N), test_Z = randm(T1,M);
//	seed = 1234234;
//	double kappa = 1;
//	matrix<double> U(T1,T1);
//	U = ML_REG_DCORR::U_ts( train_X, /* a T0 by N matrix */
//							train_Z, /* a T0 by M matrix */
//							test_X, /* a T1 by N matrix */
//							test_Z, /* a T1 by M matrix */
//							num_trees, /* number of trees */
//							feature_subsampling_fraction, /* fraction of features to consider at each node */
//							min_samples_per_leaf, /* minimum samples per leaf node */
//							kappa, /* a distance exponent */
//							seed  /* a seed */);
//
//	time = std::chrono::high_resolution_clock::now();
//
//	cout << U << endl;
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
//
//	for (int i = 0; i < T1; ++i) {
//		for (int j = 0; j < T1; ++j) {
//		test_output << U(i,j) << " , ";
//		}
//		test_output << "\n";
//	}
//
//	test_output << "Elapsed time = " << std::chrono::duration_cast<std::chrono::minutes> (time-timelast).count() << " minutes" << endl;
//	test_output.close();

//	//test the Granger causality statistic
//	double kappa = 1.0;
//	int bandwidth = 5;
//	double gc_stat = GC_TEST::gc_stat(train_X, /* a T0 by N matrix of training observations on covariates */
//										train_Y, /* a T0 by M matrix of training observations on the targets */
//										train_Z, /* a T0 by K matrix of training observations on controls */
//										test_X, /* a T1 by N matrix of test observations on covariates and controls */
//										test_Y, /* a T1 by M matrix of test observations on the targets */
//										test_Z, /* a T1 by K matrix of test observations on controls */
//										num_trees, /* number of trees */
//										feature_subsampling_fraction, /* fraction of features to consider at each node */
//										min_samples_per_leaf, /* minimum samples per leaf node */
//										kappa, /* a distance exponent */
//										bandwidth,  /* Bartlett kernel bandwidth */
//										seed  /* a seed */);
//
//	double level_sig = 0.05; // set a nominal size
//	double CV = std::pow(gsl_cdf_ugaussian_Pinv(1- (level_sig/2.0)), 2.0);
//	cout << CV << endl;
//
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
//	test_output << "Granger distance statistic = " << gc_stat << endl;
//	test_output.close();


//	gsl_rng_free(r); // free up memory

//
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
//
//	for (int i = 0; i < num_samples; ++i) {
//		test_output << (boost::format("test_label=%f, label_prediction=%f") % test_labels[i] % pred_labels[i]).str() << endl;
//	}
//	test_output.close();
//
//	random_forest_regression_trainer<dense_feature_extractor> trainer;
//	trainer.set_num_trees(num_trees);
//	trainer.set_feature_subsampling_fraction(0.5);
//	trainer.set_min_samples_per_leaf(100);
//	trainer.set_seed("random");

//	auto [train_samples, train_labels] = GenerateNoiseData(start, end, num_samples);
//	cout << train_samples[0] << endl;
//	cout << train_labels[0] << endl;
//	matrix<double> test_sample = randm(2, 1);
//	cout << "test sample = " << test_sample << endl;
//
//	double prediction = ML_REG_DCORR::RF(train_samples, /* a training sample */
//											train_labels, /* training labels */
//											test_sample, /* a test sample */
//											num_trees, /* number of trees */
//											feature_subsampling_fraction, /* fraction of features to consider at each node */
//											min_samples_per_leaf /* minimum samples per leaf node */);
//	cout << "prediction = " << prediction << endl;

//	auto random_forest = trainer.train(train_samples, train_labels);
//
//	DataSamples data_samples(num_samples);
//	Labels pred_labels(num_samples);
//	ofstream test_output;
//	test_output.open("./output.txt", ios::out);
//	size_t i = 0;
//	auto [test_samples, test_labels] = GenerateNoiseData(start, end, num_samples);
//	for (const auto& sample : test_samples) {
//		auto prediction = random_forest(sample);
//		data_samples[i] = test_labels[i];
//		pred_labels[i] = prediction;
//		test_output << (boost::format("test_label=%f, label_prediction=%f") % data_samples[i] % pred_labels[i]).str() << endl;
//		++i;
//
//	}
//	test_output.close();
//	PlotResults(data_samples, test_lables, pred_lables, "DLib Random Forest", "dlib-rf");
	//#endif




    #if 0
    gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
	unsigned long seed = 5323; //113424;
    gsl_rng_set(r, seed);


//    // Creating a directory
//    std::string name = "./Results/test_dir/";
//    const char *dirname;
//    dirname = name.c_str();
//    if (mkdir(dirname, 0777) == -1)
//        cerr << "Error :  " << strerror(errno) << endl;
//
//    else
//        cout << "Directory created" << endl;
//
    int n = 10;
	SGMatrix<double> a(n, 2), b(n, 2);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < 2; ++j) {
			a(i, j) = gsl_ran_ugaussian(r);
		}
	}
//	auto file1 = create<CSVFile>(name + "a.txt", 'w');
//	a.save(file1);
//
//	a.display_matrix("a");
//
//	gsl_ran_sample(r, b.get_column_vector(0), n, a.get_column_vector(0), n, sizeof(double) );
//	b.display_matrix("b");

	SGVector<int> index_vec(10), index_vec_bt(10);
	SGVector<int>::range_fill_vector(index_vec.vector, index_vec.vlen, 0);
	index_vec.display_vector("index_vec");
	gsl_ran_sample(r, index_vec_bt.vector, 10, index_vec.vector, 10, sizeof(int) );
	index_vec_bt.display_vector("index_vec_bt");
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 2; ++j) {
			b(i,j) = a(index_vec_bt[i], j);
		}
	}

	a.display_matrix("a");
	b.display_matrix("b");


    gsl_rng_free(r); //free up memory
    #endif



	#if 0 /** Implement the distance-based test with VAR-CC-MGARCH(1,1) **/
	gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
	unsigned long seed = 6244324;
    gsl_rng_set(r, seed);

//    int n = 100, m1 = 3, m2 = 2;
//    SGMatrix<double> A(n, m1), B(n, m2);
//    Matrix A1(n, m1), B1(n, m2);
//    for (int i = 0; i < n; ++i) {
//		for (int j = 0; j < m1; ++j) {
//			A(i,j) = gsl_rng_uniform(r);
//			A1(i+1,j+1) = A(i,j);
//		}
//    }
//
//
//
//    for (int i = 0; i < n; ++i) {
//		for (int j = 0; j < m2; ++j) {
//			B(i,j) = gsl_ran_ugaussian(r);
//			B1(i+1,j+1) = B(i,j);
//		}
//    }
//
//    int T = 200;
//    SGMatrix<double> Y1(T, 2), Y2(T, 2);
    SGVector<double> theta_var1(4), theta_var2(4), theta_var0(4);
    theta_var1[0] = 0.4;
    theta_var1[1] = 0.1;
    theta_var1[2] = -1.;
    theta_var1[3] = 0.5;
    theta_var1.display_vector("theta_var1");

    theta_var2[0] = -1.5;
    theta_var2[1] = 1.2;
    theta_var2[2] = -0.9;
    theta_var2[3] = 0.5;

	/* define CC-MGARCH parameters */
	SGVector<double> theta_mgarch1 {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	theta_mgarch1.display_vector("theta_mgarch1");

	SGVector<double> theta_mgarch2 {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	theta_mgarch2.display_vector("theta_mgarch2");

	/* define VAR-CC-MGARCH parameters */
	SGVector<double> theta_var_mgarch1 {0.4, 0.1, -1., 0.5, 0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	SGVector<double> theta_var_mgarch2 {0.4, 0.1, -0.9, 0.5, 0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
//
//    int choose_alt = 0;
//
//    NL_Dgp::gen_VAR(Y1, Y2, theta1, theta2, choose_alt, seed);
//    //Y1.display_matrix("Y1");
//    //Y2.display_matrix("Y2");
//
//    SGMatrix<double> eta1(T, 2), eta2(T,2), Y1a(T, 2);
//
//    NL_Dgp::estimate_VAR1(eta1, theta1, Y1, theta1);
//    theta1.display_vector("theta1");
//
//    SGMatrix<double> theta_mat(theta1, 2, 2);
//	theta_mat = transpose_matrix(theta_mat);
//	theta_mat.display_matrix("theta1");
//
//    NL_Dgp::estimate_VAR1(eta2, theta2, Y2, theta2);


//	Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), bootstrap_REJF(2, 1);
//	asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
//	asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
//    int number_sampl = 100;
//    int T = 200;
//    int lag_smooth = 10;
//    double expn = 1.5;
//
//    ofstream size_out;
//    size_out.open("./Results/VAR_MGARCH/size_out.txt", ios::out);
//
//	VAR_CC_MGARCH_dcorr_power::cValue<NL_Dgp::gen_VAR, daniell_kernel> (asymp_REJF, empir_CV,
//																		theta_var1, theta_var2,/*true parameters of two DGPs*/
//																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
//																									used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
//																		number_sampl, /*number of random samples drawn*/
//																		T, /*sample size*/
//																		lag_smooth, /*kernel bandwidth*/
//																		expn, /*exponent of the Euclidean distance*/
//																		asymp_CV, /*5% and 10% asymptotic critical values*/
//																		seed,
//																		size_out);
//	size_out.close();
//
//	int choose_alt = 1;
//	ofstream pwr_out;
//    pwr_out.open("./Results/VAR_MGARCH/pwr_out.txt", ios::out);
//	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, daniell_kernel>(asymp_REJF, empir_CV,
//																		theta_var1, theta_var2,/*true parameters of two DGPs*/
//																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
//																									used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
//																		number_sampl, /*number of random samples drawn*/
//																		T, /*sample size*/
//																		lag_smooth, /*kernel bandwidth*/
//																		expn, /*exponent of the Euclidean distance*/
//																		empir_CV, /*5% and 10% empirical critical values*/
//																		asymp_CV, /*5% and 10% asymptotic critical values*/
//																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
//																		seed,
//																		pwr_out);
//	pwr_out.close();
//
//
//	int num_B = 500;
//	choose_alt = 0;
//
//	![Generate two independent sequences of i.i.d. standard normal random variables]
//	SGMatrix<double> xi_x(num_B, T), xi_y(num_B, T);
//	for (auto i = 0; i < num_B; i++) {
//		for (auto t = 0; t < T; t++) {
//			xi_x(i,t) = gsl_ran_ugaussian (r);
//			xi_y(i,t) = gsl_ran_ugaussian (r);
//		}
//	}
//	pwr_out.open("./Results/VAR_MGARCH/pwr_out_boot.txt", ios::out);
//	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, daniell_kernel>(bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
//																		theta_var1, theta_var2,/*true parameters of two DGPs*/
//																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
//																																	used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
//																		number_sampl, /*number of random samples drawn*/
//																		lag_smooth, /*kernel bandwidth*/
//																		expn, /*exponent of the Euclidean distance*/
//																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
//																																							  used for bootstrapping*/
//																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
//																		seed,
//																		pwr_out);
//	pwr_out.close();

//    eta1.display_matrix("eta1");
//    eta2.display_matrix("eta2");
//
//	Y1a = NL_Dgp::gen_VAR(eta1, theta1);
//	Y1a.display_matrix("Y1a");

//    double er = 0., br = 0.;
//	std::tie(er, br) = Dep_tests::do_ElHimdiBouhaddiouiRoy<daniell_kernel>(10, 20, eta1, eta2);
//
//	cout << er << " , " << br << endl;
//
//	double L1 = 0., T1 = 0.;
//	std::tie(L1, T1) = Dep_tests::do_TchahouDuchesne(10, eta1, eta2);
//	cout << L1 << " , " << T1 << endl;

//    double rho1 = 0.8, rho4 = 0.3;
//    int T = 500;
////    SGVector<double> u1(T), u2(T);
////    SGMatrix<double> u3(T, 2), u4(T, 2);
////    std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T, rho1, rho4, seed);
////    u1.display_vector("u1");
////    u2.display_vector("u2");
////    u3.display_matrix("u3");
////    u4.display_matrix("u4");
//	SGMatrix<double> Y1(T,2), Y2(T,2);
//	int dim = 7;
//	SGVector<double> theta1(dim), theta2(dim);
//	double theta1a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
//	for (int i = 0; i < dim; ++i)
//		theta1[i] = theta1a[i];
//	theta1.display_vector("theta1");
//
//	double theta2a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
//	for (int i = 0; i < dim; ++i)
//		theta2[i] = theta2a[i];
//	theta2.display_vector("theta2");
//
//
//	int choose_alt = 1;
//    NL_Dgp::gen_CC_MGARCH(Y1, Y2, theta1, theta2, choose_alt, seed);
//    //Y1.display_matrix("Y1");
//    //Y2.display_matrix("Y2");
//
//	SGVector<double> minv(dim);
//	SGMatrix<double> resid(T, 2);
//	resid.zero();
//	double fmin = cc_mgarch::mle_simplex(resid, minv, Y1, theta1);
//
//	minv.display_vector("minv");
//	//resid.display_matrix("residuals");
//	cout << "the value of the function = " << fmin << endl;
//
//	ofstream  output;
//	output.open("minv.txt", ios::out);
//	output << std::fixed << std::setprecision(5);
//	for (int i = 0; i <= dim; ++i)
//		output << minv[i] << endl;
//	output.close();

//	double pvalue1 = 0., pvalue2 = 0.;
//	int M = 5, num_bts = 100;
//	std::tie(pvalue1, pvalue2) = HSIC::resid_bootstrap_pvalue<HSIC::kernel, HSIC::kernel,
//										cc_mgarch::mle_simplex, NL_Dgp::gen_CC_MGARCH>(Y1, Y2, theta1, theta2, M, num_bts, seed);
//	cout << "pvalues = " << pvalue1 << " , " << pvalue2 << endl;

	gsl_rng_free(r); // free up memory
	#endif





	#if 0 /***** Perform other tests *******/
	Matrix HSIC_J1_REJF(2,1), HSIC_J2_REJF(2,1), ER_asymp_REJF(2,1), ER_empir_CV(2,1), BR_asymp_REJF(2,1), BR_empir_CV(2,1), TD_L1_asymp_REJF(2,1), \
			TD_L1_empir_CV(2,1), TD_T1_asymp_REJF(2,1), TD_T1_empir_CV(2,1);

    int number_sampl = 100;
    int num_bts = 500; // number of bootstrap samples as in Wang et al. (2021)
    int T = 100;
    int lag_smooth = 10;
    double h = 0.005;
    int seed = 12425;
	bool use_bartlett_ker = false;


    /* define VAR(1) parameters */
    SGVector<double> theta1(4), theta2(4), theta0(4);
    theta1[0] = 0.4;
    theta1[1] = 0.1;
    theta1[2] = -1.;
    theta1[3] = 0.5;

    theta2[0] = -1.5;
    theta2[1] = 1.2;
    theta2[2] = -0.9;
    theta2[3] = 0.5;

    //	/* define CC-MGARCH parameters */
//    int dim = 7;
//	SGVector<double> theta1(dim), theta2(dim);
//	double theta1a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
//	for (int i = 0; i < dim; ++i)
//		theta1[i] = theta1a[i];
//	theta1.display_vector("theta1");
//
//	double theta2a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
//	for (int i = 0; i < dim; ++i)
//		theta2[i] = theta2a[i];
//	theta2.display_vector("theta2");

	/* define initial parameter values */
    int dim = 7;
	SGVector<double> theta01(dim), theta02(dim);
	double theta01a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	for (int i = 0; i < dim; ++i)
		theta01[i] = theta01a[i];
	theta01.display_vector("theta01");

	double theta02a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	for (int i = 0; i < dim; ++i)
		theta02[i] = theta02a[i];
	theta02.display_vector("theta02");

	SGMatrix<double> Y1(T,2), Y2(T,2);
	NL_Dgp::gen_CC_MGARCH(Y1, Y2, theta01, theta02, 5, seed);
//	SGVector<double> grad = cc_mgarch::neg_loglikelihood_gradient(theta01, Y1, 1e-2);
//	SGMatrix<double> hess = cc_mgarch::neg_loglikelihood_hessian(theta01, Y1, 1e-2);
//	hess.display_matrix("Hessian");
//	SGMatrix<double> hess_inv = pinv(hess);
//	hess_inv.display_matrix("Inverse Hessian");
	SGMatrix<double> ccov = cross_cov(Y1, Y2, 1);

	ccov.display_matrix("cross covariance");

	Matrix ER_asymp_CV(2,1), BR_asymp_CV(2,1), TD_L1_asymp_CV(2,1), TD_T1_asymp_CV(2,1);

	BR_asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
	BR_asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
	ER_asymp_CV = Dep_tests::asymp_CV_ChiSq3(lag_smooth);
	TD_L1_asymp_CV = Dep_tests::asymp_CV_ChiSq(lag_smooth);
	TD_T1_asymp_CV = Dep_tests::asymp_CV_ChiSq4(lag_smooth);

    ofstream size_out;
    size_out.open("./Results/Others/VAR/size_out.txt", ios::out);
//	Others_power::cValue<NL_Dgp::gen_VAR,HSIC::kernel,HSIC::kernel,cc_mgarch::mle_simplex,NL_Dgp::gen_CC_MGARCH,NL_Dgp::resid_CC_MGARCH,\
//								cc_mgarch::neg_loglikelihood_gradient, cc_mgarch::neg_loglikelihood_hessian,daniell_kernel> \
//						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
//							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
//							ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
//							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
//							BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
//							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
//							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
//							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
//							theta1, theta2, /*true parameters of two DGPs*/
//							theta01, theta02, /*initial values used to estimate DGPs*/
//							number_sampl, /*number of random samples drawn*/
//							T, /*sample size*/
//							lag_smooth, /*kernel bandwidth*/
//							num_bts, /*number of bootstrap samples*/
//							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
//							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
//							h,
//							seed,
//							size_out);

//	Others_power::cValue<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,cc_mgarch::mle_simplex,NL_Dgp::gen_CC_MGARCH,NL_Dgp::resid_CC_MGARCH,\
//								cc_mgarch::neg_loglikelihood_gradient, cc_mgarch::neg_loglikelihood_hessian,daniell_kernel> \
//						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
//							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
//							ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
//							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
//							BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
//							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
//							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
//							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
//							theta01, theta02, /*true parameters of two DGPs*/
//							theta01, theta02, /*initial values used to estimate DGPs*/
//							number_sampl, /*number of random samples drawn*/
//							T, /*sample size*/
//							lag_smooth, /*kernel bandwidth*/
//							num_bts, /*number of bootstrap samples*/
//							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
//							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
//							h,
//							seed,
//							size_out,
//							use_bartlett_ker);

//	Others_power::cValue<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel, NL_Dgp::estimate_VAR1, NL_Dgp::gen_VAR, NL_Dgp::resid_VAR1,\
//								NL_Dgp::sse_VAR1_gradient, NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
//						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
//							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
//							ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
//							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
//							BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
//							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
//							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
//							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
//							theta01, theta02, /*true parameters of two DGPs*/
//							theta1, theta2, /*initial values used to estimate DGPs*/
//							number_sampl, /*number of random samples drawn*/
//							T, /*sample size*/
//							lag_smooth, /*kernel bandwidth*/
//							num_bts, /*number of bootstrap samples*/
//							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
//							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
//							h,
//							seed,
//							size_out);
//

	size_out.close();

	Matrix HSIC_J1_empir_REJF(2,1), HSIC_J2_empir_REJF(2,1), ER_empir_REJF(2,1), BR_empir_REJF(2,1), TD_L1_empir_REJF(2,1), TD_T1_empir_REJF(2,1);

	int choose_alt = 1;

	ofstream pwr_out;
    pwr_out.open("./Results/Others/VAR/pwr_out.txt", ios::out);

//	Others_power::power_f<NL_Dgp::gen_VAR,HSIC::kernel,HSIC::kernel,cc_mgarch::mle_simplex,NL_Dgp::gen_CC_MGARCH,NL_Dgp::resid_CC_MGARCH,\
//								cc_mgarch::neg_loglikelihood_gradient, cc_mgarch::neg_loglikelihood_hessian,daniell_kernel> \
//							(HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
//							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
//							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
//							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
//							theta1, theta2, /*true parameters of two DGPs*/
//							theta01, theta02, /*initial values used to estimate DGPs*/
//							number_sampl, /*number of random samples drawn*/
//							T, /*sample size*/
//							lag_smooth, /*kernel bandwidth*/
//							num_bts, /*number of bootstrap samples*/
//							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
//							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
//							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
//							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
//							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
//							choose_alt, /*set a degree of dependence between two DGPs*/
//							h, /*a finite differential factor*/
//							seed, /*seed for random number generator*/
//							pwr_out);

//	Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,cc_mgarch::mle_simplex,NL_Dgp::gen_CC_MGARCH,NL_Dgp::resid_CC_MGARCH,\
//								cc_mgarch::neg_loglikelihood_gradient, cc_mgarch::neg_loglikelihood_hessian,daniell_kernel> \
//							(HSIC_J1_empir_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
//							HSIC_J2_empir_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
//							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
//							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
//							theta01, theta02, /*true parameters of two DGPs*/
//							theta01, theta02, /*initial values used to estimate DGPs*/
//							number_sampl, /*number of random samples drawn*/
//							T, /*sample size*/
//							lag_smooth, /*kernel bandwidth*/
//							num_bts, /*number of bootstrap samples*/
//							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
//							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
//							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
//							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
//							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
//							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
//							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
//							choose_alt, /*set a degree of dependence between two DGPs*/
//							h, /*a finite differential factor*/
//							seed, /*seed for random number generator*/
//							pwr_out,
//							use_bartlett_ker);

//	Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel, NL_Dgp::estimate_VAR1, NL_Dgp::gen_VAR, NL_Dgp::resid_VAR1,\
//									NL_Dgp::sse_VAR1_gradient, NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
//								(HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
//								HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
//								ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
//								BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
//								TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
//								TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
//								theta01, theta02, /*true parameters of two DGPs*/
//								theta1, theta2, /*initial values used to estimate DGPs*/
//								number_sampl, /*number of random samples drawn*/
//								T, /*sample size*/
//								lag_smooth, /*kernel bandwidth*/
//								num_bts, /*number of bootstrap samples*/
//								ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
//								ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
//								BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
//								BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
//								TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
//								TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
//								TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
//								TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
//								choose_alt, /*set a degree of dependence between two DGPs*/
//								h, /*a finite differential factor*/
//								seed, /*seed for random number generator*/
//								pwr_out);

	pwr_out.close();
	#endif









    #if 0 /****** Perform the distance-based test ********/
    gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, 143523);

//    int n = 50, n_test = 50, dim = 20;
//    double rho = 0.8, sigma = 0.5; // set the AR coefficient and the noise sigma
//    SGMatrix<double> x(n, dim), x_test(n_test, dim);
//    SGVector<double> y(n), y_test(n_test);
//    y.zero();
//    for (int i = 0; i < n; ++i) {
//		for (int j = 0; j < dim; ++j) {
//			if (i == 0) {
//				x(i, j) = gsl_ran_gaussian(r, 1.);
//			}
//			else {
//				x(i, j) = rho * x(i-1, j) + gsl_ran_gaussian(r, sigma);
//			}
//			y[i] += x(i, j) + gsl_ran_gaussian(r, sigma); //+ pow(x(i, j), 2.);
//		}
//    }
//	y.display_vector();
//	cout << "\n" << endl;
//
//	y_test.zero();
//    for (int i = 0; i < n_test; ++i) {
//		for (int j = 0; j < dim; ++j) {
//			if (i == 0) {
//				x_test(i, j) = gsl_ran_gaussian(r, 1.);
//			}
//			else {
//				x_test(i, j) = rho * x_test(i-1, j) + gsl_ran_gaussian(r, sigma);
//			}
//			y_test[i] += x_test(i, j) + gsl_ran_gaussian(r, sigma); //+ pow(x(i, j), 2.);
//		}
//    }
//

	//GBM_Plot(x, y, x_test, y_test, 5, 200, 0.08, 0.6, 12424);

	//RF_Plot(x, y, x_test, y_test, int(dim/2), 20, 1243252);

	//ML_REG::SVR_Plot(x, y, x_test, y_test, 0.1, 1.0, 1.0, 0.1, 2);
	//ML_REG::LARS_Plot(x, y, x_test, y_test, 0.01);
	//ML_REG::KRR_Plot(x, y, x_test, y_test, 1., 0.001);

//	int seed = 63432;
//	int L = 2;
//	int lag_smooth = 7;
//	double expn = 1.5;
//
//	int num_subsets = 1, min_subset_size = 10, tree_max_depths_list_size = 2, num_iters_list_size = 1, learning_rates_list_size = 10, \
//		subset_fractions_list_size = 4, num_rand_feats_list_size = L, num_bags_list_size = 3;
//	SGVector<int> tree_max_depths_list(tree_max_depths_list_size), num_iters_list(num_iters_list_size), \
//					num_rand_feats_list(num_rand_feats_list_size), num_bags_list(num_bags_list_size);
//	SGVector<double> learning_rates_list(learning_rates_list_size), subset_fractions_list(subset_fractions_list_size);
//
//	for (int i = 0; i < tree_max_depths_list_size; ++i)
//		tree_max_depths_list[i] = 4*(i+1);
//	tree_max_depths_list.display_vector("tree_max_depths_list");
//
//	for (int i = 0; i < num_iters_list_size; ++i)
//		num_iters_list[i] = i*100 + 200;
//	num_iters_list.display_vector("num_iters_list");
//
//	for (int i = 0; i < learning_rates_list_size; ++i)
//		learning_rates_list[i] =  0.01*(i+1);
//	learning_rates_list.display_vector("learning_rates_list");
//
//	for (int i = 0; i < subset_fractions_list_size; ++i)
//		subset_fractions_list[i] = 0.1*(i+1);
//	subset_fractions_list.display_vector("subset_fractions_list");
//
//	for (int i = 0; i < num_rand_feats_list_size; ++i)
//		num_rand_feats_list[i] = 2*(i + 1);
//	num_rand_feats_list.display_vector("number of features list");
//
//	for (int i = 0; i < num_bags_list_size; ++i)
//		num_bags_list[i] = 10*i + 15;
//	num_bags_list.display_vector("number of bags list");

//	int opt_tree_max_depth = 0, opt_num_iters = 0, opt_num_rand_feats = 0., opt_num_bags = 0.;
//	double opt_learning_rate = 0, opt_subset_fraction = 0;
//
//	std::tie(opt_tree_max_depth, opt_num_iters, opt_num_rand_feats, opt_num_bags, opt_learning_rate, opt_subset_fraction) =
//			ML_REG::RF_cv(x, y, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, learning_rates_list, subset_fractions_list, \
//							num_rand_feats_list, num_bags_list, 14535);
//
//	SGVector<double> new_labels = ML_REG::RF_Plot(x, y, opt_tree_max_depth, opt_num_iters, opt_num_rand_feats, \
//														opt_num_bags, opt_learning_rate, opt_subset_fraction, 1342);
//
//	SGMatrix<double> output_mat(n, 2);
//	output_mat.set_column(0, y);
//	output_mat.set_column(1, new_labels);
//	output_mat.display_matrix("output_mat");




	int T = 100;
    SGMatrix<double> Y1(T, 2), Y2(T, 2);

//    /* define VAR(1) parameters */
//    SGVector<double> theta1(4), theta2(4);
//    theta1[0] = 0.4;
//    theta1[1] = 0.1;
//    theta1[2] = -1.;
//    theta1[3] = 0.5;
//
//    theta2[0] = -1.5;
//    theta2[1] = 1.2;
//    theta2[2] = -0.9;
//    theta2[3] = 0.5;

	/* define CC-MGARCH parameters */
    int dim = 7;
	SGVector<double> theta1(dim), theta2(dim);
	double theta1a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	for (int i = 0; i < dim; ++i)
		theta1[i] = theta1a[i];
	theta1.display_vector("theta1");

	double theta2a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	for (int i = 0; i < dim; ++i)
		theta2[i] = theta2a[i];
	theta2.display_vector("theta2");



//	NL_Dgp::gen_CC_MGARCH(Y1, Y2, theta1, theta2, 5, seed);
//	Y1.display_matrix("Y1");
//	Y2.display_matrix("Y2");


//	auto file1 = create<CSVFile>("./Results/ML/Y1.csv", 'w');
//	auto file2 = create<CSVFile>("./Results/ML/Y2.csv", 'w');
//
//    int choose_alt = 0;
//    int n = 20;
//    SGMatrix<double> big_Y1(T, n), big_Y2(T, n);
//    for (int i = 0; i < n; ++i) {
//		NL_Dgp::gen_CC_MGARCH(Y1, Y2, theta1, theta2, choose_alt, i+1);
//		big_Y1.set_column( i, Y2.get_column(0) );
//		big_Y2.set_column( i, Y2.get_column(1) );
//    }
//    transpose_matrix(big_Y1).save(file1);
//    transpose_matrix(big_Y2).save(file2);


//	SGMatrix<double> mat_reg_first(T-L, T-L), mat_reg_second(T-L, T-L), mat_breg(T-L, T-L);
//	ML_REG_DCORR::reg<ML_REG::RF_cv, ML_REG::RF_Plot> (mat_reg_first, mat_reg_second, mat_breg, Y1, L, expn,
//						num_subsets, /*number of subsets for TSCV*/
//						min_subset_size, /*minimum subset size for TSCV*/
//						tree_max_depths_list, /*list of tree max depths (for GBM)*/
//						num_iters_list, /*list of numbers of iterations (for GBM)*/
//						learning_rates_list, /*list of learning rates (for GBM)*/
//						subset_fractions_list, /*list of subset fractions (for GBM)*/
//						num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
//						num_bags_list, /*list of number of bags (for RF)*/
//						seed);
//	cout << "size of 'mat_reg' = " << mat_breg.num_rows << " , " << mat_breg.num_cols << endl;
//	// FILE *fd = fopen ( "/etc/resolv.conf" , "r" );
//	// fclose ( fd );
//	auto file1 = create<CSVFile>("./mat_reg1.csv", 'w');
//	auto file2 = create<CSVFile>("./mat_reg2.csv", 'w');
//	auto file3 = create<CSVFile>("./mat_breg.csv", 'w');
//	mat_reg_first.save(file1);
//	mat_reg_second.save(file2);
//	mat_breg.save(file3);


//    double kernel_QDSum, kernel_QRSum;
//	double stat = ML_DCORR::do_Test<ML_REG::RF_cv, ML_REG::RF_Plot, daniell_kernel> (	Y1, Y2, L, lag_smooth, expn,
//																						num_subsets, /*number of subsets for TSCV*/
//																						min_subset_size, /*minimum subset size for TSCV*/
//																						tree_max_depths_list, /*list of tree max depths (for GBM)*/
//																						num_iters_list, /*list of numbers of iterations (for GBM)*/
//																						learning_rates_list, /*list of learning rates (for GBM)*/
//																						subset_fractions_list, /*list of subset fractions (for GBM)*/
//																						num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
//																						num_bags_list, /*list of number of bags (for RF)*/
//																						kernel_QDSum, kernel_QRSum,
//																						seed);
//	cout << "The value of the ML dtest = " << stat << endl;



	Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1);
	asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
	asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
    int number_sampl = 300;

//    ofstream size_out;
//    size_out.open("./Results/ML/VAR/size_out.txt", ios::out);
//	ML_dcorr_power::cValue<NL_Dgp::gen_VAR, ML_DCORR::do_Test<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>, bartlett_kernel> (asymp_REJF, empir_CV,
//								theta1, theta2, /*true parameters of two DGPs*/
//								number_sampl, /*number of random samples drawn*/
//								T, /*sample size*/
//								L, /*maximum truncation lag*/
//								lag_smooth, /*kernel bandwidth*/
//								expn, /*exponent of the Euclidean distance*/
//								asymp_CV, /*5% and 10% asymptotic critical values*/
//								num_subsets, /*number of subsets for TSCV*/
//								min_subset_size, /*minimum subset size for TSCV*/
//								tree_max_depths_list, /*list of tree max depths (for GBM)*/
//								num_iters_list, /*list of numbers of iterations (for GBM)*/
//								learning_rates_list, /*list of learning rates (for GBM)*/
//								subset_fractions_list, /*list of subset fractions (for GBM)*/
//								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
//								num_bags_list, /*list of number of bags (for RF)*/
//								seed,
//								size_out);
//	size_out.close();
//
//	ofstream pwr_out;
//
//	int choose_alt = 1;
//	while (choose_alt <= 5) {
//		pwr_out.open("./Results/ML/VAR/pwr_out_" + std::to_string(choose_alt) + ".txt", ios::out);
//		ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>, bartlett_kernel> (empir_REJF, asymp_REJF,
//								theta1, theta2, /*true parameters of two DGPs*/
//								number_sampl, /*number of random samples drawn*/
//								T, /*sample size*/
//								L, /*maximum truncation lag*/
//								lag_smooth, /*kernel bandwidth*/
//								expn, /*exponent of the Euclidean distance*/
//								empir_CV, /*5% and 10% empirical critical values*/
//								asymp_CV, /*5% and 10% asymptotic critical values*/
//								num_subsets, /*number of subsets for TSCV*/
//								min_subset_size, /*minimum subset size for TSCV*/
//								tree_max_depths_list, /*list of tree max depths (for GBM)*/
//								num_iters_list, /*list of numbers of iterations (for GBM)*/
//								learning_rates_list, /*list of learning rates (for GBM)*/
//								subset_fractions_list, /*list of subset fractions (for GBM)*/
//								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
//								num_bags_list, /*list of number of bags (for RF)*/
//								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
//								seed,
//								pwr_out);
//		pwr_out.close();
//
//		choose_alt += 1;
//	}

//	int choose_alt = 0;
//	Matrix bootstrap_REJF(2,1);
//	int num_B = 500;
//
//	//![Generate two independent sequences of i.i.d. standard normal random variables]
//	SGMatrix<double> xi_x(num_B, T), xi_y(num_B, T);
//	for (auto i = 0; i < num_B; i++) {
//		for (auto t = 0; t < T; t++) {
//			xi_x(i,t) = gsl_ran_ugaussian (r);
//			xi_y(i,t) = gsl_ran_ugaussian (r);
//		}
//	}
//
//	ofstream pwr_out;
//	while (choose_alt <= 5) {
//		pwr_out.open("./Results/ML/VAR/Bootstrap/pwr_out_" + std::to_string(choose_alt) + ".txt", ios::out);
//
//		ML_dcorr_power::power_f<NL_Dgp::gen_CC_MGARCH, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>> \
//							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
//								theta1, theta2, /*true parameters of two DGPs*/
//								number_sampl, /*number of random samples drawn*/
//								L, /*maximum truncation lag*/
//								lag_smooth, /*kernel bandwidth*/
//								expn, /*exponent of the Euclidean distance*/
//								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
//								num_subsets, /*number of subsets for TSCV*/
//								min_subset_size, /*minimum subset size for TSCV*/
//								tree_max_depths_list, /*list of tree max depths (for GBM)*/
//								num_iters_list, /*list of numbers of iterations (for GBM)*/
//								learning_rates_list, /*list of learning rates (for GBM)*/
//								subset_fractions_list, /*list of subset fractions (for GBM)*/
//								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
//								num_bags_list, /*list of number of bags (for RF)*/
//								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
//								seed,
//								pwr_out);
//
//		pwr_out.close();
//
//		choose_alt++;
//	}



	//auto file1 = create<CSVFile>("./mat_U.csv", 'w');
	//mat_U.save(file1);

//	int target_dim = 2;
//	SGMatrix<double> pc_train(n, target_dim), pc_test(n_test, target_dim);
//	std::tie(pc_train, pc_test) = ML_REG::calcul_PCs(x, x_test, target_dim);


//	int num_subsets = 4, min_subset_size = 30, kernel_widths_list_size = 20, C1_list_size = 20, C2_list_size = 20, tube_epsilons_list_size = 3;
//	SGVector<double> kernel_widths_list(kernel_widths_list_size), C1_list(C1_list_size), C2_list(C2_list_size), \
//																		tube_epsilons_list(tube_epsilons_list_size);
//	for (int i = 0; i < kernel_widths_list_size; ++i)
//		kernel_widths_list[i] = 0.1 + 0.5*i;
//
//	for (int i = 0; i < C1_list_size; ++i)
//		C1_list[i] = 0.1 + 0.5*i;
//	for (int i = 0; i < C2_list_size; ++i)
//		C2_list[i] = 0.1 + 0.5*i;
//	for (int i = 0; i < tube_epsilons_list_size; ++i)
//		tube_epsilons_list[i] = 0.05 + 0.1*i;
//
//	ML_REG::SVRegression(x, y, x_test, y_test, num_subsets, min_subset_size, kernel_widths_list, C1_list, C2_list, tube_epsilons_list);



//	int num_subsets = 4, min_subset_size = 30, num_rand_feats_list_size = 10, num_bags_list_size = 10;
//	SGVector<int> num_rand_feats_list(num_rand_feats_list_size), num_bags_list(num_bags_list_size);
//	for (int i = 0; i < num_rand_feats_list_size; ++i)
//		num_rand_feats_list[i] = i + 5;
//	num_rand_feats_list.display_vector("number of features list");
//
//	for (int i = 0; i < num_bags_list_size; ++i)
//		num_bags_list[i] = i + 10;
//	num_bags_list.display_vector("number of bags list");
//
//	int seed = 1432532;
//	ML_REG::RFRegression(x, y, x_test, y_test, num_subsets, min_subset_size, num_rand_feats_list, num_bags_list, seed);

//	int num_subsets = 4, min_subset_size = 5, tree_max_depths_list_size = 2, num_iters_list_size = 1, learning_rates_list_size = 10, \
//		subset_fractions_list_size = 4;
//	SGVector<int> tree_max_depths_list(tree_max_depths_list_size), num_iters_list(num_iters_list_size);
//	SGVector<double> learning_rates_list(learning_rates_list_size), subset_fractions_list(subset_fractions_list_size);
//
//	for (int i = 0; i < tree_max_depths_list_size; ++i)
//		tree_max_depths_list[i] = 4*(i+1);
//	tree_max_depths_list.display_vector("tree_max_depths_list");
//
//	for (int i = 0; i < num_iters_list_size; ++i)
//		num_iters_list[i] = i*100 + 200;
//	num_iters_list.display_vector("num_iters_list");
//
//	for (int i = 0; i < learning_rates_list_size; ++i)
//		learning_rates_list[i] =  0.01*(i+1);
//	learning_rates_list.display_vector("learning_rates_list");
//
//	for (int i = 0; i < subset_fractions_list_size; ++i)
//		subset_fractions_list[i] = 0.1*(i+1);
//	subset_fractions_list.display_vector("subset_fractions_list");
//
//	int seed = 13425;
//	GBMRegression(x, y, x_test, y_test, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, learning_rates_list, \
//					subset_fractions_list, seed);


//	int num_subsets = 4, min_subset_size = 5, num_iters_min = 100, num_iters_max = 400, tree_max_depth = 6;
//	double learning_rate_min = 0.01, learning_rate_max = 0.1, subset_fraction_min = 0.6, subset_fraction_max = 1;
//	int seed = 1234324;
//
//	GBMRegressionCV(x, y, x_test, y_test, num_subsets, min_subset_size, num_iters_min, num_iters_max,
//					tree_max_depth, learning_rate_min, learning_rate_max, subset_fraction_min, subset_fraction_max, seed);



	gsl_rng_free(r);
	#endif




    #if 0
    // generate data
	const size_t seed = 3463;
	const size_t num_samples = 20;
	int dim = 7;
	SGMatrix<DataType> x_values;
	SGVector<DataType> y_values;
	std::tie(x_values, y_values) = GenerateShogunData(-10, 10, num_samples, dim, seed);

	int num_subsets_int = 4;
	int min_subset_size_int = 5;
	SGMatrix<int> train_indices, test_indices;
    std::tie(train_indices, test_indices) = tscv(y_values, num_subsets_int, min_subset_size_int);
    train_indices.display_matrix();
    cout << "\n" << endl;
    test_indices.display_matrix();
    #endif

    #if 0
    //init_shogun_with_defaults();
  // shogun::sg_io->set_loglevel(shogun::MSG_INFO);

	// generate data
	const size_t seed = 3463;
	const size_t num_samples = 500;
	int dim = 7;
	SGMatrix<DataType> x_values;
	SGVector<DataType> y_values;
	std::tie(x_values, y_values) = GenerateShogunData(-10, 10, num_samples, dim, seed);

	auto train_features = some<CDenseFeatures<DataType>>(x_values);
	SGMatrix<DataType>  feature_matrix;
	feature_matrix = train_features->get_feature_matrix();
	// feature_matrix.display_matrix();
	cout << x_values.num_rows << " , " << x_values.num_cols << endl;

	auto train_labels = some<CRegressionLabels>(y_values);
	cout << y_values.size() << endl;


	std::tie(x_values, y_values) = GenerateShogunData(-10, 10, num_samples, dim, seed);
	auto test_features = some<CDenseFeatures<DataType>>(x_values);
	auto test_labels = some<CRegressionLabels>(y_values);

	GBMClassification(train_features, train_labels, test_features, test_labels, dim);
	RFClassification(train_features, train_labels, test_features, test_labels, dim);

	//exit_shogun();
    #endif

    #if 0
    // initialize
    //init_shogun_with_defaults();

    // create some data
    SGMatrix<float64_t> matrix(2,3);
    for(int i=0; i<2; i++) {
        for (int j = 0; j < 3; j++) {
            matrix(i, j) = i + j;
        }
    }
    matrix.display_matrix();

    // create three 2-dimensional vectors
    CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>();
    features->set_feature_matrix(matrix);

    // create three labels
    CBinaryLabels* labels = new CBinaryLabels(3);
    labels->set_label(0, -1);
    labels->set_label(1, +1);
    labels->set_label(2, -1);

    // create gaussian kernel(RBF) with cache 10MB, width 0.5
    CGaussianKernel* kernel = new CGaussianKernel(10, 0.5);
    kernel->init(features, features);

    // create libsvm with C=10 and train
    CLibSVM* svm = new CLibSVM(10, kernel, labels);
    svm->train();

    SG_SPRINT("total sv:%d, bias:%f\n", svm->get_num_support_vectors(), svm->get_bias());

    // classify on training examples
    for(int i=0; i<3; i++) {
    SG_SPRINT("output[%d]=%f\n", i, svm->apply_one(i));
    }

    // free up memory
    SG_UNREF(svm);

    //exit_shogun();

    #endif





    /*gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, 143523);
    int T, T_max, d_x, d_y, number_sampl, nblocks, bsize, bandwidth, seed;
    double alpha = 1., ddep, pwr = 0.;
    number_sampl = 500;
    T = 50;
    T_max = 500;
    d_x = 4;
    d_y = 4;
    nblocks = 300;
    bandwidth = 5;
    bsize = 30;
    ddep = 0.;*/


    #if 0
    unsigned long seed = 134235235;
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    int T = 200;
    int N_x = 3, N_y = 4;
	int poly_degree = 4, nbreak = 5, L_T = 2;
	int ncoeffs = nbreak + poly_degree - 2;
	int T0 = T- L_T;
    Matrix X(T,N_x), Y(T,N_y), beta_x(N_x*L_T*ncoeffs,N_x), beta_y(N_y*L_T*ncoeffs,N_y);

    //generate X
    X(1,1) = gsl_ran_gaussian (r, 0.5);
    X(1,3) = gsl_ran_gaussian(r, 1.5);
    for (auto t = 2; t <= T; ++t) {
        X(t,1) = 0.8*X(t-1,1) + gsl_ran_ugaussian (r);
        X(t,3) = 0.1*sin(X(t-1,3)) + gsl_ran_ugaussian(r);
    }
    X(1,2) = gsl_ran_gaussian (r, 0.5);
    X(2,2) = gsl_ran_gaussian (r, 1.);
	for (auto t = 3; t <= T; ++t) {
		//X(t,2) = 0.8*log(1+3*pow(X(t-1,2), 2.)) - 0.6*log(1+3*pow(X(t-2,2), 2.)) + 1.5*sin(M_PI_2*X(t-1,1)) - 1.*sin(M_PI_4*X(t-2,1)) + gsl_ran_ugaussian(r);
		X(t,2) = 0.8*log(1+3*pow(X(t-1,2), 2.)) - 0.6*log(1+3*pow(X(t-2,2), 2.)) - 1.*sin(M_PI_4*X(t-2,1)) + gsl_ran_ugaussian(r);
	}

	Matrix X2(T,1);
	for (auto t = 1; t <= T; ++t)
		X2(t) = X(t,2);

	//generate Y
	Y(1,1) = gsl_ran_gaussian (r, 0.5);
	Y(1,3) = gsl_ran_gaussian (r, 1.5);
	Y(1,4) = gsl_ran_gaussian (r, 0.8);
	for (auto t = 2; t <= T; ++t) {
		Y(t,1) = 0.2*Y(t-1,1) + gsl_ran_ugaussian (r);
		Y(t,3) = 0.1*cos(Y(t-1,3)) + gsl_ran_ugaussian(r);
          Y(t,4) = 0.7*exp(-fabs(Y(t-1,4))) + gsl_ran_ugaussian(r);
	}
	Y(1,2) = gsl_ran_gaussian (r, 0.5);
	Y(2,2) = gsl_ran_gaussian (r, 1.);
	for (auto t = 3; t <= T; ++t) {
		Y(t,2) = 0.8*log(1+3*pow(Y(t-1,2), 2.)) - 0.6*log(1+3*pow(Y(t-2,2), 2.)) + 1.5*sin(M_PI_2*Y(t-1,1)) - 1.*sin(M_PI_4*Y(t-2,1)) + gsl_ran_ugaussian(r);
	}

	do_Norm(X); //normalize data to the 0-1 range
	do_Norm(Y);

	Matrix X0(T0,N_x), Y0(T0,N_y);
	for (auto t = L_T + 1; t <= T; ++t) {
		for (auto i = 1; i <= N_x; ++i)
			X0(t-L_T,i) = X(t,i);
	}
	for (auto t = L_T+1; t <= T; ++t) {
		for (auto i = 1; i <= N_y; ++i)
			Y0(t-L_T,i) = Y(t,i);
	}

	//assign initial values for all the slope parameters
	Matrix beta_init(N_x*L_T*ncoeffs,1);
	for (auto i = 1; i <= N_x*L_T*ncoeffs; ++i) {
		beta_init(i) = gsl_ran_ugaussian(r);
		for (auto j = 1; j <= N_x; ++j) {
			beta_x(i,j) = gsl_ran_ugaussian(r);
		}
	}
	for (auto i = 1; i <= N_y*L_T*ncoeffs; ++i) {
		for (auto j = 1; j <= N_y; ++j) {
			beta_y(i,j) = gsl_ran_ugaussian(r);
		}
	}


     gsl_bspline_workspace *bw; //define workspace for B-splines
     bw = gsl_bspline_alloc(poly_degree, nbreak);
     //use uniform breakpoints on [0, 1]
     gsl_bspline_knots_uniform(0.0, 1.0, bw);
    //calculate the knots
	//for (auto k = 1; k <= ncoeffs+poly_degree; ++k )
        //cout << "knot = " << k <<  ": " << gsl_vector_get(bw->knots, k-1) << endl;
     Matrix BS_x(N_x*L_T*ncoeffs,T0), BS_y(N_y*L_T*ncoeffs,T0), ave_BS_x(N_x*L_T*ncoeffs,1), ave_BS_y(N_y*L_T*ncoeffs,1);
     gsl_vector *B = gsl_vector_alloc (ncoeffs);
     for (auto  j= 1; j <= N_x; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = L_T+1; t <= T; ++t) {
                         gsl_bspline_eval(X(t-ell,j), B, bw); //evaluate all B-spline basis functions at X(t-ell,j)
                         for (auto k = 1; k <= ncoeffs; ++k) {
						BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t-L_T) = poly_degree * gsl_vector_get(B, k-1) / (nbreak-1); //normalize all the B-splines basis functions
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
                         }
               }
          }
     }
      for (auto  j= 1; j <= N_y; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = L_T+1; t <= T; ++t) {
                         gsl_bspline_eval(Y(t-ell,j), B, bw); //evaluate all B-spline basis functions at Y(t-ell,j)
                         for (auto k = 1; k <= ncoeffs; ++k) {
						BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t-L_T) = poly_degree * gsl_vector_get(B, k-1) / (nbreak-1); //normalize all the B-splines basis functions
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
                         }
               }
          }
     }
     gsl_bspline_free (bw); //free memory
     gsl_vector_free (B);
     ave_BS_x.set(0.);
     for (auto  j= 1; j <= N_x; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k) {
				for (auto t = 1; t <= T0; ++t) {
					ave_BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) += BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) / T0; //calculate the temporal averages for all B-spline basis functions (BS_x)
				}
			}
          }
     }
     for (auto  j= 1; j <= N_x; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = 1; t <= T0; ++t) {
				for (auto k = 1; k <= ncoeffs; ++k) {
					//calculate the centered B-spline basis functions for BS_x
					BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) - ave_BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
					//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
				}
               }
          }
     }
      ave_BS_y.set(0.);
     for (auto  j= 1; j <= N_y; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k) {
				for (auto t = 1; t <= T0; ++t) {
					ave_BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) += BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) / T0; //calculate the temporal averages for all B-spline basis functions (BS_x)
				}
			}
          }
     }
     for (auto  j= 1; j <= N_y; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = 1; t <= T0; ++t) {
				for (auto k = 1; k <= ncoeffs; ++k) {
					//calculate the centered B-spline basis functions for BS_x
					BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) - ave_BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
					//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
				}
               }
          }
     }



	/*int max_iter = 500, ngrid = 5;
	auto min_tol = 1e-4, lambda_glasso_rate = pow(T0*log(N_x*L_T*ncoeffs), 0.5), opt_BIC = 0., opt_EBIC = 0., lb = 0.2, ub = 2.;
	//Matrix opt_tuning(2,1);
	auto lambda1 = pow(T0*log(N_x*L_T*ncoeffs), 0.5), lambda2 = pow(T0, 0.5);
	ofstream  glasso_out, ic_out;
	glasso_out.open ("lasso.txt", ios::out);
	ic_out.open ("ic.txt", ios::out);
	VAR_gLASSO::calcul_adap_gLasso (beta_init, X2, X, L_T, poly_degree, nbreak, lambda1, lambda2, max_iter, min_tol, glasso_out);
	//opt_tuning = VAR_gLASSO::calcul_IC (opt_BIC, opt_EBIC, beta_init, Y, BS, lambda_glasso_rate, lb, ub, ngrid, N, L_T, max_iter, min_tol, ic_out, glasso_out);
	glasso_out.close();
	ic_out.close();*/


	ofstream  reg_out;
	reg_out.open ("euclidean_reg1.txt", ios::out);
	int lag_smooth = 5;
	auto stat = NGDist_corr::do_Test< bartlett_kernel> (X0, Y0, BS_x, BS_y, beta_x, beta_y, lag_smooth, 1.5);
	cout << "the value of the test statistic = " << stat << endl;
	reg_out.close();
     gsl_rng_free (r); //free memory

     #endif

	#if 0
     //compare two data matrices
	ifstream inStream1;//import the first dataset
	inStream1.open ("out_stat1.txt", ios::in);
    if (!inStream1)
        return(EXIT_FAILURE);
    vector<vector<string>* > data1;
    loadCSV(inStream1, data1);
    int T = data1.size(); //Rows
    int N = (*data1[1]).size(); //Columns
    Matrix _X(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            _X(t,i) = hex2d((*data1[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p1 = data1.begin( ); p1 != data1.end( ); ++p1)
    {
      delete *p1;
    }
    inStream1.close();

    ifstream inStream2;//import the second dataset
	inStream2.open ("out_stat.txt", ios::in);
    if (!inStream2)
        return(EXIT_FAILURE);
    vector<vector<string>* > data2;
    loadCSV(inStream2, data2);
    T = data2.size(); //Rows
    N = (*data2[1]).size(); //Columns
    Matrix _Y(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            _Y(t,i) = hex2d((*data2[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p2 = data2.begin( ); p2 != data2.end( ); ++p2)
    {
      delete *p2;
    }
    inStream2.close();
    bool res = true;
    res = compare(_X,_Y);
    cout << "res = " << res << endl;
    #endif


    # if 0
    int TL = 2;
    int T = 100;
	int number_sampl = 500;
    Matrix X(T,1), Y(T,1), x1(TL,2), x2(TL,2), x0(1,2);
    Matrix alpha(2,1), beta(2,1), lambda(2,1), sigma(2,1);
    alpha(1) = 0.01;
    alpha(2) = 0.04;
    beta(1) = 0.7;
    beta(2) = 0.4;
    lambda(1) = 0.;
    lambda(2) = -1;
	sigma(1) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h
	sigma(2) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h
	double rho = 1.54 * sigma(1); // set rho equal to 1.54 for zero correlation
    double cdf = 0.95;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 20;
    double cv = 0.;
    Matrix reject_rate(2, 1);
    output_filename = "./Results/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
	size_filename = "./Results/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    power_filename = "./Results/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    Power obj_power;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //*****************************************************************************************************************************************//
    lag_smooth = 10;
    output_filename = "./Results/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
	size_filename = "./Results/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    power_filename = "./Results/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //***************************************************************************************************************************************//
    lag_smooth = 5;
    output_filename = "./Results/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
	size_filename = "./Results/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    power_filename = "./Results/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
    # endif

	# if 0
	 int TL = 2;
     int T = 100;
     int N = 500;
	 int num_B = 1000;
	 double expn = 1.5, rho12 = -0.5, rho13 = 0.2;
	 int choose_alt = 1;
     unsigned long seed = 9436434;
     int lag_smooth = 5; //set M_T = 5, 10, 20, and 25
     Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), alpha_X(5,1), alpha_Y(5,2), epsilon_t(N,1), epsilon_s(N,1), eta1_t(N,1), eta1_s(N,1), eta2_t(N,1),
                 eta2_s(N,1), delta(3,1);
     asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
     asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
     alpha_X(1) = 0.01; //define the AR coefficients
     alpha_X(2) = 0.7;
     alpha_X(3) = 0.1;
     alpha_X(4) = -0.6;
     alpha_X(5) = 0.3;
     alpha_Y(1,1) = 0.04;
     alpha_Y(2,1) = 0.4;
     alpha_Y(3,1) = 0.5;
     alpha_Y(4,1) = 0.1;
     alpha_Y(5,1) = -0.6;
     alpha_Y(1,2) = 0.01;
     alpha_Y(2,2) = -0.2;
     alpha_Y(3,2) = 0.1;
     alpha_Y(4,2) = 0.05;
     alpha_Y(5,2) = 0.4;

     delta(1) = 0.1;
     delta(2) = 0.9;
     delta(3) = -0.1;
     NL_Dgp::gen_RANV<NL_Dgp::gen_TriSN> (epsilon_t, eta1_t, eta2_t, delta, 0., 0., 0, 13435); //generate independent skewed-normal random errors
     NL_Dgp::gen_RANV<NL_Dgp::gen_TriSN> (epsilon_s, eta1_s, eta2_s, delta, 0., 0., 0, 53435);

	int size_alpha_X = alpha_X.nRow(), size_alpha_Y = alpha_Y.nRow();
	Matrix X(T,1), Y1(T,1), Y2(T,1), alpha_X_hat(size_alpha_X,1), alpha_Y1_hat(size_alpha_Y,1), alpha_Y2_hat(size_alpha_Y,1), alpha_Y_hat(size_alpha_Y,2),
			  resid_X(T-2,1), resid_Y1(T-2,1), resid_Y2(T-2,1);

     NL_Dgp::gen_TAR<NL_Dgp::gen_TriSN>  (X, Y1, Y2, alpha_X, alpha_Y, delta, rho12, rho13, choose_alt, seed); //generate data

	//then use these samples to estimate the AR coefficients
	NL_Dgp::est_TAR (resid_X, alpha_X_hat, X);
	NL_Dgp::est_TAR (resid_Y1, alpha_Y1_hat, Y1);
	NL_Dgp::est_TAR (resid_Y2, alpha_Y2_hat, Y2);
	for (int j = 1; j <= size_alpha_Y; j++) {
		alpha_Y_hat(j,1) = alpha_Y1_hat(j); //collect all the OLS estimates in a matrix
		alpha_Y_hat(j,2) = alpha_Y2_hat(j);
	}

	Matrix cstat_bootstrap(num_B, 1);
	double cstat = 0.;
	Matrix xi_x(num_B, T), xi_y(num_B, T);
	gsl_rng *r = nullptr;
     const gsl_rng_type *gen; //random number generator
     gsl_rng_env_setup();
     gen = gsl_rng_taus;
     r = gsl_rng_alloc(gen);
     gsl_rng_set(r, 145325);
	for (auto i = 1; i <= num_B; i++) {
			for (auto t = 1; t <= T; t++) {
				xi_x(i,t) = gsl_ran_ugaussian (r);
				xi_y(i,t) = gsl_ran_ugaussian (r);
			}
	}
	gsl_rng_free (r); //free memory

	auto pvalue = NGDist_corr::calcul_Pvalue <bartlett_kernel, NGReg::cmean_TAR> (cstat, cstat_bootstrap, X, Y1, Y2, TL, lag_smooth, alpha_X_hat, alpha_Y_hat,
																									epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t, eta2_s, xi_x, xi_y, expn);

	ofstream  bootstrap;
	bootstrap.open ("boot_stat.txt", ios::out);
	for (int i = 1; i <= num_B; i++)
			bootstrap << cstat_bootstrap(i) << endl;
	bootstrap << "the value of the statistics = " << cstat << endl;
	cout << "p-value = " << pvalue<< endl;
	bootstrap << "p-value = " << pvalue<< endl;
     bootstrap.close();
     # endif

	# if 0
     ifstream inStream1;//import the first dataset
	inStream1.open ("E:\\Copy\\SCRIPTS\\Exogeneity\\Application\\data\\GT\\Features\\returns.csv", ios::in);
    if (!inStream1)
        return(EXIT_FAILURE);
    vector<vector<string>* > data1;
    loadCSV(inStream1, data1);
    int T = data1.size(); //Rows
    int N = (*data1[1]).size(); //Columns
    Matrix X(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            X(t,i) = hex2d((*data1[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p1 = data1.begin( ); p1 != data1.end( ); ++p1)
    {
      delete *p1;
    }
    inStream1.close();

    ifstream inStream2;//import the second dataset
	inStream2.open ("E:\\Copy\\SCRIPTS\\Exogeneity\\Application\\data\\GT\\Features\\cmeans.csv", ios::in);
    if (!inStream2)
        return(EXIT_FAILURE);
    vector<vector<string>* > data2;
    loadCSV(inStream2, data2);
    T = data2.size(); //Rows
    N = (*data2[1]).size(); //Columns
    Matrix cmean(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            cmean(t,i) = hex2d((*data2[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p1 = data2.begin( ); p1 != data2.end( ); ++p1)
    {
      delete *p1;
    }
    inStream2.close();

    ifstream inStream3;//import the first dataset
	inStream3.open ("E:\\Copy\\SCRIPTS\\Exogeneity\\Application\\data\\GT\\Features\\ccovs.csv", ios::in);
    if (!inStream3)
        return(EXIT_FAILURE);
     vector<vector<string>* > data3;
    loadCSV(inStream3, data3);
    T = data3.size(); //Rows
    N = (*data3[1]).size(); //Columns
    Matrix csigma(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            csigma(t,i) = hex2d((*data3[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p1 = data3.begin( ); p1 != data3.end( ); ++p1)
    {
      delete *p1;
    }
    inStream3.close();

    ifstream inStream4;//import the first dataset
	inStream4.open ("E:\\Copy\\SCRIPTS\\Exogeneity\\Application\\data\\GT\\Features\\residuals.csv", ios::in);
    if (!inStream4)
        return(EXIT_FAILURE);
    vector<vector<string>* > data4;
    loadCSV(inStream4, data4);
    T = data4.size(); //Rows
    N = (*data4[1]).size(); //Columns
    Matrix resid(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            resid(t,i) = hex2d((*data4[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p1 = data4.begin( ); p1 != data4.end( ); ++p1)
    {
      delete *p1;
    }
    inStream4.close();

    Matrix reg0(T, T);

    NGReg::var_U (reg0, X, cmean, csigma, resid, 1.5);

    ofstream  output;
	output.open ("out_stat.txt", ios::out);
	for (int i = 1; i <= T; i++) {
		for (int j = 1; j <= T; j++) {
			output << reg0(i, j) << " , ";
		}
		output << " \n ";
	}
     output.close();
	# endif


    //please do not comment out the lines below.
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    time = std::chrono::high_resolution_clock::now();
    //output << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //cout << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //output << "This program took " << std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count() << " seconds to run.\n";
    auto duration =  std::chrono::duration_cast <std::chrono::milliseconds> (time-timelast).count();
    cout << "This program took " << duration << " seconds (" << duration << " milliseconds) to run." << endl;
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    //system("PAUSE");
    return 0;
}






