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
#include <power3.h>



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

	matrix<double> mu1(2,1); /* means */
	mu1 = 16.5/52,
		  16.5/52;

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
	string ci_filename, ci_lens_filename;

	ofstream ci_lens_out, ci_out;
	ci_lens_out << std::fixed << std::setprecision(4);
	ci_out << std::fixed << std::setprecision(4);

	// Creating a directory
	dir_name = "./Results/LW/argarch/";
	if (std::filesystem::create_directories(dir_name)) {
		std::cout << "Directory tree created successfully: " << dir_name << std::endl;
	} else {
		std::cout << "Failed to create directory tree or it already exists." << std::endl;
	}

	matrix<double> ci_lens_t_stat_mean(bws.nr(),2), ci_lens_boot_t_stat_mean(bws.nr(),Ls.nr()+2), ci_lens_sn_stat_mean(bws.nr(),1),
					ci_lens_mean_99(bws.nr(),Ls.nr()+3), ci_lens_mean_95(bws.nr(),Ls.nr()+3), ci_lens_mean_90(bws.nr(),Ls.nr()+3);
	matrix<double> ci_lens_t_stat_std(bws.nr(),2), ci_lens_boot_t_stat_std(bws.nr(),Ls.nr()+2), ci_lens_sn_stat_std(bws.nr(),1),
					ci_lens_std_99(bws.nr(),Ls.nr()+3), ci_lens_std_95(bws.nr(),Ls.nr()+3), ci_lens_std_90(bws.nr(),Ls.nr()+3);
	while (T <= 750) {
		cout << "T = " << T << endl;
		cout << "mu = \n" << mu1 << endl;

		/* Calculate sizes */
		ci_filename = dir_name + "ci_lens_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + ".txt";
		ci_out.open(ci_filename, ios::out);
		ci_out << "T = " << T << endl;

		ci_out << "kernel bandwidths = " << endl;
		ci_out << bws << endl;

		ci_out << "Block sizes = " << endl;
		ci_out << Ls << endl;

		ci_out << "A = \n" << A << endl;
		ci_out << "B = \n" << B << endl;
		ci_out << "C = \n" << C << endl;
		ci_out << "mu = \n" << mu1 << endl;

		ci_out << "Calculate CI lengths ..." << endl;


		auto [ci_lens_t_stat_mean_99, ci_lens_t_stat_std_99, ci_lens_t_stat_mean_95, ci_lens_t_stat_std_95, ci_lens_t_stat_mean_90, ci_lens_t_stat_std_90,
			  ci_lens_boot_t_stat_mean_99, ci_lens_boot_t_stat_std_99, ci_lens_boot_t_stat_mean_95, ci_lens_boot_t_stat_std_95, ci_lens_boot_t_stat_mean_90,
			  ci_lens_boot_t_stat_std_90,
			  ci_lens_sn_stat_mean_99, ci_lens_sn_stat_std_99, ci_lens_sn_stat_mean_95, ci_lens_sn_stat_std_95, ci_lens_sn_stat_mean_90, ci_lens_sn_stat_std_90]
											= Power::ci_lens<QS_kernel,
															 Dgp::gen_dvech_argarch<gsl_ran_gaussian>>(num_samples, /* number of random samples */
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
																										ci_out, /* output stream */
																										seed /* a seed to generate random numbers */);
		/* ==================================================================================================================================================== */
		ci_lens_filename = dir_name + "ci_lens_mean_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_99" + ".txt";
		ci_lens_out.open(ci_lens_filename, ios::out);
		ci_lens_sn_stat_mean = ci_lens_sn_stat_mean_99;
		ci_lens_t_stat_mean = join_rows(bws, ci_lens_t_stat_mean_99);
		ci_lens_boot_t_stat_mean = join_rows(ci_lens_t_stat_mean, ci_lens_boot_t_stat_mean_99);
		ci_lens_mean_99 = join_rows(ci_lens_boot_t_stat_mean, ci_lens_sn_stat_mean);
		Dgp::print_matrix_csv<double>(ci_lens_mean_99, ci_lens_out);
		ci_lens_out.close();
		/* ===================================================================================================================================================== */
		ci_lens_filename = dir_name + "ci_lens_std_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_99" + ".txt";
		ci_lens_out.open(ci_lens_filename, ios::out);
		ci_lens_sn_stat_std = ci_lens_sn_stat_std_99;
		ci_lens_t_stat_std = join_rows(bws, ci_lens_t_stat_std_99);
		ci_lens_boot_t_stat_std = join_rows(ci_lens_t_stat_std, ci_lens_boot_t_stat_std_99);
		ci_lens_std_99 = join_rows(ci_lens_boot_t_stat_std, ci_lens_sn_stat_std);
		Dgp::print_matrix_csv<double>(ci_lens_std_99, ci_lens_out);
		ci_lens_out.close();
		/* ===================================================================================================================================================== */
		/* ==================================================================================================================================================== */
		ci_lens_filename = dir_name + "ci_lens_mean_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_95" + ".txt";
		ci_lens_out.open(ci_lens_filename, ios::out);
		ci_lens_sn_stat_mean = ci_lens_sn_stat_mean_95;
		ci_lens_t_stat_mean = join_rows(bws, ci_lens_t_stat_mean_95);
		ci_lens_boot_t_stat_mean = join_rows(ci_lens_t_stat_mean, ci_lens_boot_t_stat_mean_95);
		ci_lens_mean_95 = join_rows(ci_lens_boot_t_stat_mean, ci_lens_sn_stat_mean);
		Dgp::print_matrix_csv<double>(ci_lens_mean_95, ci_lens_out);
		ci_lens_out.close();
		/* ===================================================================================================================================================== */
		ci_lens_filename = dir_name + "ci_lens_std_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_95" + ".txt";
		ci_lens_out.open(ci_lens_filename, ios::out);
		ci_lens_sn_stat_std = ci_lens_sn_stat_std_95;
		ci_lens_t_stat_std = join_rows(bws, ci_lens_t_stat_std_95);
		ci_lens_boot_t_stat_std = join_rows(ci_lens_t_stat_std, ci_lens_boot_t_stat_std_95);
		ci_lens_std_95 = join_rows(ci_lens_boot_t_stat_std, ci_lens_sn_stat_std);
		Dgp::print_matrix_csv<double>(ci_lens_std_95, ci_lens_out);
		ci_lens_out.close();
		/* ==================================================================================================================================================== */
		ci_lens_filename = dir_name + "ci_lens_mean_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_90" + ".txt";
		ci_lens_out.open(ci_lens_filename, ios::out);
		ci_lens_sn_stat_mean = ci_lens_sn_stat_mean_90;
		ci_lens_t_stat_mean = join_rows(bws, ci_lens_t_stat_mean_90);
		ci_lens_boot_t_stat_mean = join_rows(ci_lens_t_stat_mean, ci_lens_boot_t_stat_mean_90);
		ci_lens_mean_90 = join_rows(ci_lens_boot_t_stat_mean, ci_lens_sn_stat_mean);
		Dgp::print_matrix_csv<double>(ci_lens_mean_90, ci_lens_out);
		ci_lens_out.close();
		/* ===================================================================================================================================================== */
		ci_lens_filename = dir_name + "ci_lens_std_err_dist=" + err_dist + "_kernel=" + kernel + "_T=" + std::to_string(T) + "_90" + ".txt";
		ci_lens_out.open(ci_lens_filename, ios::out);
		ci_lens_sn_stat_std = ci_lens_sn_stat_std_90;
		ci_lens_t_stat_std = join_rows(bws, ci_lens_t_stat_std_90);
		ci_lens_boot_t_stat_std = join_rows(ci_lens_t_stat_std, ci_lens_boot_t_stat_std_90);
		ci_lens_std_90 = join_rows(ci_lens_boot_t_stat_std, ci_lens_sn_stat_std);
		Dgp::print_matrix_csv<double>(ci_lens_std_90, ci_lens_out);
		ci_lens_out.close();
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
