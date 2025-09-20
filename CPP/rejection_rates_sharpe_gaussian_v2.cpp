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
#include <cassert>
#include <map>
#include <filesystem>
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

// #include <kernel.h>
// #include <tests3.h>
// #include <dgp.h>
// #include <sharpe_ratios.h>
// //#include <bootstraps.h>
// #include <test_sharpe_ratios.h>
// #include <sn_test_sharpe_ratios.h>
#include <power2.h>



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

	matrix<double> bws(6,1); /* kernel bandwidths */
//	bws = 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70;
	bws = 1, 5, 10, 15, 20, 25;
	matrix<int> Ls(10,1); /* block sizes */
	Ls = 10, 20, 30, 40, 50, 60, 70, 80, 90, 100;

	const int num_samples = 1000; /* number of random samples */
	const int num_boots = 499; /* number of bootstrap repetitions */
	int T = 150; /* sample size */
	const double nu = 1.; /* parameter of the error distribution */

	matrix<double> mu1(2,1), mu2(2,1), mu3(2,1); /* means */
	mu1 = 16.5/52,
		  16.5/52;
	mu2 = 16.6/52,
		  15.6/52;
	mu3 = 16.6/52,
		  10.6/52;

	matrix<double> A(2,2), B(2,2), C(2,2); /* GARCH coefficients */
	A = 0.075, 0.050,
		0.050, 0.075;
	B = 0.90, 0.89,
		0.89, 0.90;
	C = 0.15, 0.13,
		0.13, 0.15;

	cout << A << endl;
	cout << B << endl;
	cout << C << endl;

	const string err_dist = "Gaussian";
	const string kernel = "QS_kernel";
	unsigned long seed = 123456789;

	string dir_name = "";
	string stats_filename, size_filename, pwr_filename;

	ofstream size_out, pwr_out, stats_out;
	size_out << std::fixed << std::setprecision(4);
	pwr_out << std::fixed << std::setprecision(4);
	stats_out << std::fixed << std::setprecision(4);

	// Creating a directory
	dir_name = "./Results/LW/bgaussian/";
	if (std::filesystem::create_directories(dir_name)) {
		std::cout << "Directory tree created successfully: " << dir_name << std::endl;
	} else {
		std::cout << "Failed to create directory tree or it already exists." << std::endl;
	}

	matrix<double> sn_REJF_vec(bws.nr(),1);
	matrix<double> REJF1a(2,1), REJF1b(Ls.nr()+2,1), REJF1c(Ls.nr()+3,1), REJF1(Ls.nr()+4,1);
	matrix<double> REJF5a(2,1), REJF5b(Ls.nr()+2,1), REJF5c(Ls.nr()+3,1), REJF5(Ls.nr()+4,1);
	matrix<double> REJF10a(2,1), REJF10b(Ls.nr()+2,1), REJF10c(Ls.nr()+3,1), REJF10(Ls.nr()+4,1);
	while (T <= 750) {
		cout << "T = " << T << endl;
		cout << "mu = \n" << mu1 << endl;

		/* Calculate sizes */
		stats_filename = dir_name + "stats_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + ".txt";
		stats_out.open(stats_filename, ios::out);
		stats_out << "T = " << T << endl;

		stats_out << "kernel bandwidths = " << endl;
		stats_out << bws << endl;

		stats_out << "Block sizes = " << endl;
		stats_out << Ls << endl;

		stats_out << "A = \n" << A << endl;
		stats_out << "B = \n" << B << endl;
		stats_out << "C = \n" << C << endl;
		stats_out << "Calculate size ..." << endl;
		stats_out << "mu = \n" << mu1 << endl;


		auto [asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10]
											= Power::power_f<QS_kernel,
															Dgp::gen_bgaussian>(num_samples, /* number of random samples */
																				T, /* sample size */
																				bws, /* kernel bandwidths */
																				Ls, /* block sizes */
																				num_boots, /* number of bootstrap repetitions */
																				nu, /* parameter of the error distribution */
																				mu1, /* means */
																				A, /* ARCH coefficients */
																				B, /* AR coefficients */
																				C, /* intercepts */
																				err_dist, /* distribution of the errors */
																				stats_out, /* output stream */
																				seed /* a seed to generate random numbers */);
		/* ==================================================================================================================================================== */
		size_filename = dir_name + "size_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_1" + ".txt";
		size_out.open(size_filename, ios::out);
		sn_REJF_vec = sn_REJF1;
		REJF1a = join_rows(bws, asymp_REJF1);
		REJF1b = join_rows(REJF1a, boot_REJF1);
		REJF1c = join_rows(REJF1b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF1;
		REJF1 = join_rows(REJF1c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF1, size_out);
		size_out.close();
		/* ===================================================================================================================================================== */
		size_filename = dir_name + "size_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_5" + ".txt";
		size_out.open(size_filename, ios::out);
		sn_REJF_vec = sn_REJF5;
		REJF5a = join_rows(bws, asymp_REJF5);
		REJF5b = join_rows(REJF5a, boot_REJF5);
		REJF5c = join_rows(REJF5b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF5;
		REJF5 = join_rows(REJF5c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF5, size_out);
		size_out.close();
		/* ===================================================================================================================================================== */
		size_filename = dir_name + "size_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_10" + ".txt";
		size_out.open(size_filename, ios::out);
		sn_REJF_vec = sn_REJF10;
		REJF10a = join_rows(bws, asymp_REJF10);
		REJF10b = join_rows(REJF10a, boot_REJF10);
		REJF10c = join_rows(REJF10b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF10;
		REJF10 = join_rows(REJF10c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF10, size_out);
		size_out.close();
		/* ===================================================================================================================================================== */

		/* Calculate power */
		stats_out << "Calculate power ..." << endl;
		stats_out << "mu = \n" << mu2 << endl;
		std::tie(asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10)
											= Power::power_f<QS_kernel,
															Dgp::gen_bgaussian>(num_samples, /* number of random samples */
																				T, /* sample size */
																				bws, /* kernel bandwidths */
																				Ls, /* block sizes */
																				num_boots, /* number of bootstrap repetitions */
																				nu, /* parameter of the error distribution */
																				mu2, /* means */
																				A, /* ARCH coefficients */
																				B, /* AR coefficients */
																				C, /* intercepts */
																				err_dist, /* distribution of the errors */
																				stats_out, /* output stream */
																				seed /* a seed to generate random numbers */);
		/* ==================================================================================================================================================== */
		pwr_filename = dir_name + "pwr_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_1_" + "mu2.txt";
		pwr_out.open(pwr_filename, ios::out);
		sn_REJF_vec = sn_REJF1;
		REJF1a = join_rows(bws, asymp_REJF1);
		REJF1b = join_rows(REJF1a, boot_REJF1);
		REJF1c = join_rows(REJF1b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF1;
		REJF1 = join_rows(REJF1c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF1, pwr_out);
		pwr_out.close();
		/* ===================================================================================================================================================== */
		pwr_filename = dir_name + "pwr_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_5_" + "mu2.txt";
		pwr_out.open(pwr_filename, ios::out);
		sn_REJF_vec = sn_REJF5;
		REJF5a = join_rows(bws, asymp_REJF5);
		REJF5b = join_rows(REJF5a, boot_REJF5);
		REJF5c = join_rows(REJF5b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF5;
		REJF5 = join_rows(REJF5c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF5, pwr_out);
		pwr_out.close();
		/* ===================================================================================================================================================== */
		pwr_filename = dir_name + "pwr_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_10_" + "mu2.txt";
		pwr_out.open(pwr_filename, ios::out);
		sn_REJF_vec = sn_REJF10;
		REJF10a = join_rows(bws, asymp_REJF10);
		REJF10b = join_rows(REJF10a, boot_REJF10);
		REJF10c = join_rows(REJF10b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF10;
		REJF10 = join_rows(REJF10c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF10, pwr_out);
		pwr_out.close();
		/* ===================================================================================================================================================== */


		stats_out << "Calculate power ..." << endl;
		stats_out << "mu = \n" << mu3 << endl;
		std::tie(asymp_REJF1, asymp_REJF5, asymp_REJF10, boot_REJF1, boot_REJF5, boot_REJF10, sn_REJF1, sn_REJF5, sn_REJF10, sn2_REJF1, sn2_REJF5, sn2_REJF10)
											= Power::power_f<QS_kernel,
															Dgp::gen_bgaussian>(num_samples, /* number of random samples */
																				T, /* sample size */
																				bws, /* kernel bandwidths */
																				Ls, /* block sizes */
																				num_boots, /* number of bootstrap repetitions */
																				nu, /* parameter of the error distribution */
																				mu3, /* means */
																				A, /* ARCH coefficients */
																				B, /* AR coefficients */
																				C, /* intercepts */
																				err_dist, /* distribution of the errors */
																				stats_out, /* output stream */
																				seed /* a seed to generate random numbers */);
		stats_out.close();
		/* ==================================================================================================================================================== */
		pwr_filename = dir_name + "pwr_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_1_" + "mu3.txt";
		pwr_out.open(pwr_filename, ios::out);
		sn_REJF_vec = sn_REJF1;
		REJF1a = join_rows(bws, asymp_REJF1);
		REJF1b = join_rows(REJF1a, boot_REJF1);
		REJF1c = join_rows(REJF1b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF1;
		REJF1 = join_rows(REJF1c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF1, pwr_out);
		pwr_out.close();
		/* ===================================================================================================================================================== */
		pwr_filename = dir_name + "pwr_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_5_" + "mu3.txt";
		pwr_out.open(pwr_filename, ios::out);
		sn_REJF_vec = sn_REJF5;
		REJF5a = join_rows(bws, asymp_REJF5);
		REJF5b = join_rows(REJF5a, boot_REJF5);
		REJF5c = join_rows(REJF5b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF5;
		REJF5 = join_rows(REJF5c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF5, pwr_out);
		pwr_out.close();
		/* ===================================================================================================================================================== */
		pwr_filename = dir_name + "pwr_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_10_" + "mu3.txt";
		pwr_out.open(pwr_filename, ios::out);
		sn_REJF_vec = sn_REJF10;
		REJF10a = join_rows(bws, asymp_REJF10);
		REJF10b = join_rows(REJF10a, boot_REJF10);
		REJF10c = join_rows(REJF10b, sn_REJF_vec);
		sn_REJF_vec = sn2_REJF10;
		REJF10 = join_rows(REJF10c, sn_REJF_vec);
		Dgp::print_matrix_csv<double>(REJF10, pwr_out);
		pwr_out.close();
		/* ===================================================================================================================================================== */

		T += 100;
	}

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
