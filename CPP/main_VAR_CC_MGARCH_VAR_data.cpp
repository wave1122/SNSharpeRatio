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

#include <matrix_ops2.h>
#include <dist_corr.h>
#include <kernel.h>
#include <nl_dgp.h>
#include <dep_tests.h>
#include <plot.h>
#include <utils.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/config.h>
//#include <shogun/base/init.h>
//#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/util/iterators.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/linop/MatrixOperator.h>

#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>


//#include <ShogunML/ml_reg_6_1_4.h>
#include <ShogunML/data/data.h>
#include <VAR_CC_MGARCH_dcorr_power.h>


#define CHUNK 1



using namespace std;
using namespace shogun;
using namespace shogun::linalg;


int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;


	//#if 0 //begin comment
	/***************************************************** Start the Simulations for the Proposed Test ********************************************************************/
    /************************************************** True DGP: VAR(1) and Fitted DGP: VAR-CC-MGARCH(1,1) ***************************************************************/
    unsigned long int seed = 1535;
    int T = 100;
	int lag_smooth = 7; // set M_T = 3, 5, 7, 10, 20, and 25
	double expn = 1.5;
	int number_sampl = 500;
	int num_B = 500;


	/* define VAR parameters */
	SGVector<double> theta_var1(4), theta_var2(4);
    theta_var1[0] = 0.4;
    theta_var1[1] = 0.1;
    theta_var1[2] = -1.;
    theta_var1[3] = 0.5;
    theta_var1.display_vector("theta_var1");

    theta_var2[0] = -1.5;
    theta_var2[1] = 1.2;
    theta_var2[2] = -0.9;
    theta_var2[3] = 0.5;
    theta_var2.display_vector("theta_var2");

	/* define CC-MGARCH parameters */
	SGVector<double> theta_mgarch1 {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	theta_mgarch1.display_vector("theta_mgarch1");

	SGVector<double> theta_mgarch2 {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	theta_mgarch2.display_vector("theta_mgarch2");

//	/* define VAR-CC-MGARCH parameters */
//	SGVector<double> theta_var_mgarch1 {0.4, 0.1, -1., 0.5, 0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
//	SGVector<double> theta_var_mgarch2 {-1.5, 1.2, -0.9, 0.5, 0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};

	//![Generate two independent sequences of i.i.d. standard normal random variables]
	gsl_rng *r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    SGMatrix<double> xi_x(num_B, T), xi_y(num_B, T);
	for (auto i = 0; i < num_B; i++) {
		for (auto t = 0; t < T; t++) {
			xi_x(i,t) = gsl_ran_ugaussian (r);
			xi_y(i,t) = gsl_ran_ugaussian (r);
		}
	}

	gsl_rng_free(r); // free up memory


	const char *new_dir;
	string dir_name, output_filename, size_filename, power_filename;
	ofstream output, size_out, pwr_out;

	Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), bootstrap_REJF(2, 1);
	asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
	asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
	int choose_alt = 0;

	output << std::fixed << std::setprecision(5);

    /************************************************************ Using the Bartlett kernel ***********************************************************************/
	// Creating a directory
    dir_name = "./Results/VAR_CC_MGARCH/VAR/BL/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
    else
        cout << "Directory created" << endl;

    output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_bartlett_kernel_var_data.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_bartlett_kernel_var_data.txt";
    power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_bartlett_kernel_var_data.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;

	choose_alt = 0;

	size_out << "====================================================== Compute sizes ==================================================" << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
	VAR_CC_MGARCH_dcorr_power::cValue<NL_Dgp::gen_VAR, bartlett_kernel> (asymp_REJF, empir_CV,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2, /*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		seed,
																		size_out);

    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5% - and 10% - bootstrap sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5% - and 10% - bootstrap sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

    size_out << "======================================================== Done with calculation of sizes ====================================================" << endl;

    choose_alt = 1;  //correlation (between eta1 and eta2) = 0.3

    pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, bartlett_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	choose_alt = 2;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, bartlett_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;


	choose_alt = 3;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, bartlett_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;


	choose_alt = 4;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, bartlett_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	choose_alt = 5;  //the errors (eta1 and eta2) are highly correlated
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, bartlett_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	output.close();
    size_out.close();
    pwr_out.close();

    /***************************************************************** Using the Daniell kernel ***********************************************************************/
	// Creating a directory
    dir_name = "./Results/VAR_CC_MGARCH/VAR/DN/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
    else
        cout << "Directory created" << endl;

    output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_daniell_kernel_var_data.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_daniell_kernel_var_data.txt";
    power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_daniell_kernel_var_data.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;

	choose_alt = 0;

	size_out << "====================================================== Compute sizes ==================================================" << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
	VAR_CC_MGARCH_dcorr_power::cValue<NL_Dgp::gen_VAR, daniell_kernel> (asymp_REJF, empir_CV,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2, /*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		seed,
																		size_out);

    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5% - and 10% - bootstrap sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5% - and 10% - bootstrap sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

    size_out << "======================================================== Done with calculation of sizes ====================================================" << endl;

    choose_alt = 1;  //correlation (between eta1 and eta2) = 0.3

    pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, daniell_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	choose_alt = 2;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, daniell_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;


	choose_alt = 3;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, daniell_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;


	choose_alt = 4;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, daniell_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	choose_alt = 5;  //the errors (eta1 and eta2) are highly correlated
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, daniell_kernel>(empir_REJF, asymp_REJF,
																		bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																		theta_var1, theta_var2,/*true parameters of two DGPs*/
																		theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																												used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																		number_sampl, /*number of random samples drawn*/
																		T, /*sample size*/
																		lag_smooth, /*kernel bandwidth*/
																		expn, /*exponent of the Euclidean distance*/
																		empir_CV, /*5% and 10% empirical critical values*/
																		asymp_CV, /*5% and 10% asymptotic critical values*/
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																										  used for bootstrapping*/
																		choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																		seed,
																		pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the QS kernel ***********************************************************************/
	// Creating a directory
    dir_name = "./Results/VAR_CC_MGARCH/VAR/QS/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
    else
        cout << "Directory created" << endl;

    output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_QS_kernel_var_data.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_QS_kernel_var_data.txt";
    power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_QS_kernel_var_data.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;

	choose_alt = 0;

	size_out << "====================================================== Compute sizes ==================================================" << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
	VAR_CC_MGARCH_dcorr_power::cValue<NL_Dgp::gen_VAR, QS_kernel> (	asymp_REJF, empir_CV,
																	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																	theta_var1, theta_var2, /*true parameters of two DGPs*/
																	theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																	number_sampl, /*number of random samples drawn*/
																	T, /*sample size*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	asymp_CV, /*5% and 10% asymptotic critical values*/
																	xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																									  used for bootstrapping*/
																	seed,
																	size_out);

    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5% - and 10% - bootstrap sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5% - and 10% - bootstrap sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

    size_out << "======================================================== Done with calculation of sizes ====================================================" << endl;

    choose_alt = 1;  //correlation (between eta1 and eta2) = 0.3

    pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, QS_kernel>(	empir_REJF, asymp_REJF,
																	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																	theta_var1, theta_var2,/*true parameters of two DGPs*/
																	theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																	number_sampl, /*number of random samples drawn*/
																	T, /*sample size*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	empir_CV, /*5% and 10% empirical critical values*/
																	asymp_CV, /*5% and 10% asymptotic critical values*/
																	xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																									  used for bootstrapping*/
																	choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																	seed,
																	pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	choose_alt = 2;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, QS_kernel>(	empir_REJF, asymp_REJF,
																	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																	theta_var1, theta_var2,/*true parameters of two DGPs*/
																	theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																	number_sampl, /*number of random samples drawn*/
																	T, /*sample size*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	empir_CV, /*5% and 10% empirical critical values*/
																	asymp_CV, /*5% and 10% asymptotic critical values*/
																	xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																									  used for bootstrapping*/
																	choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																	seed,
																	pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;


	choose_alt = 3;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, QS_kernel>(	empir_REJF, asymp_REJF,
																	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																	theta_var1, theta_var2,/*true parameters of two DGPs*/
																	theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																	number_sampl, /*number of random samples drawn*/
																	T, /*sample size*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	empir_CV, /*5% and 10% empirical critical values*/
																	asymp_CV, /*5% and 10% asymptotic critical values*/
																	xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																									  used for bootstrapping*/
																	choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																	seed,
																	pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;


	choose_alt = 4;  //the errors (eta1 and eta2) are not correlated, but dependent
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, QS_kernel>(	empir_REJF, asymp_REJF,
																	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																	theta_var1, theta_var2,/*true parameters of two DGPs*/
																	theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																	number_sampl, /*number of random samples drawn*/
																	T, /*sample size*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	empir_CV, /*5% and 10% empirical critical values*/
																	asymp_CV, /*5% and 10% asymptotic critical values*/
																	xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																									  used for bootstrapping*/
																	choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																	seed,
																	pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	choose_alt = 5;  //the errors (eta1 and eta2) are highly correlated
	 pwr_out << "===================================================== Compute power =================================================" << endl;
	VAR_CC_MGARCH_dcorr_power::power_f<NL_Dgp::gen_VAR, QS_kernel>(	empir_REJF, asymp_REJF,
																	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
																	theta_var1, theta_var2,/*true parameters of two DGPs*/
																	theta_mgarch1, theta_mgarch2, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
																	number_sampl, /*number of random samples drawn*/
																	T, /*sample size*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	empir_CV, /*5% and 10% empirical critical values*/
																	asymp_CV, /*5% and 10% asymptotic critical values*/
																	xi_x, xi_y, /*num_B by T matrices of auxiliary random variables
																																									  used for bootstrapping*/
																	choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
																	seed,
																	pwr_out);


	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
	cout << "5%- and 10% - bootstrap rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	pwr_out << "====================================================== Done with power calculation =============================================================" << endl;

	output.close();
    size_out.close();
    pwr_out.close();

	/***************************************** Finish the Simulations for the Proposed Test *********************************************************************/
    /************************************************************************************************************************************************************/
	//#endif //end comment


	#if 0 //begin comment
	/* Calculate empirical levels of a test */
    //Import a column of data
    ifstream data;
    data.open("e:/Dropbox/Codes/Supermicro-Office-1/Exogeneity/Results/HimdiRoy/size_out_T=200_lag_smooth=25_threshold=1.54_bivariate.txt", ios::in);
    if (!data)
        return(EXIT_FAILURE);
    vector<string> data1;
    loadCSV(data, data1);
    int N = data1.size(); //Rows
    Matrix X(N, 1);
    for (auto i = 1; i <= N; i++)
        X(i) = hex2d(data1[i-1]);
    data.close();
    int count10 = 0, count5 = 0;
    Dep_tests dep_obj;
    Matrix cv(2, 1);
    int bandW = 25; //set a lag-smoothing parameter
    cv = dep_obj.asymp_CV_ChiSq2 (bandW);
    for (auto i = 1; i <= N; ++i) {
    	if (X(i) >= cv(2)) ++count10;
    	if (X(i) >= cv(1)) ++count5;
	}
	cout << "N = " << N << endl;
	cout << "10% rejection rate = " << ((double) count10/N) << endl;
    cout << "5% rejection rate = " << ((double) count5/N) << endl;
    /*//Import a csv file
    ifstream data;
    data.open("./Results/Others/Correlation/QS/size_out_T=200_lag_smooth=25_rho=0.80000000000000004.txt", ios::in);
    if (!data)
        return(EXIT_FAILURE);
    vector<vector<string>* > data1;
    loadCSV(data, data1);
    int nR = data1.size(); //Rows
    int nC = (*data1[1]).size(); //Columns
    Matrix X(nR,nC);
    for (auto i = 1; i <= nR; i++)
    {
        for (int j = 1; j <= nC; j++)
        {
            X(i,j) = hex2d((*data1[i-1])[j-1]);
        }
    }
    for (vector<vector<string>*>::iterator p = data1.begin( ); p != data1.end( ); ++p)
    {
        delete *p;
    }
    data.close();
    Matrix count10(nC, 1), count5(nC, 1), asymp_cv_Haugh(2, 1);
    int bandW = 25; //set a lag-smoothing parameter
    Dep_tests dep_obj;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    for (auto i = 1; i <= nR; ++i) {
    	if (X(i,1) >= 1.28) ++count10(1);
    	if (X(i,1) >= 1.645) ++count5(1);
    	if (X(i,2) >= asymp_cv_Haugh(2)) ++count10(2);
    	if (X(i,2) >= asymp_cv_Haugh(1)) ++count5(2);
	}
	cout << "N = " << nR << endl;
	for (auto j = 1; j <= nC; ++j) {
	    cout << "Test = " << j << ": 10% rejection rate = " << ((double) count10(j)/nR) << endl;
        cout << "Test = " << j << ": 5% rejection rate = " << ((double) count5(j)/nR) << endl;
    }*/
    #endif //end comment

    #if 0 //begin comment
    /************************************** Start the Simulations for the Proposed Test: The Univariate Skewed-Normal Case **************************************/
    /************************************************************************************************************************************************************/
	int TL = 2;
    int T = 100;
    int N = 1000;
	int number_sampl = 500;
	double expn = 1.5, delta = 0.7, rho = 0.8;
	int choose_alt = 0;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 5; //set M_T = 5, 10, 20, and 25
    //============================================================= Bilinear Models ===============================================================================/
    Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), alpha_X(6,1), alpha_Y(6,1), epsilon_t(N,1), epsilon_s(N,1), eta_t(N,1), eta_s(N,1);
    asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
    asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
    alpha_X(1) = 0.01; //define the AR coefficients
    alpha_X(2) = 0.5;
    alpha_X(3) = -0.2;
    alpha_X(4) = -0.1;
    alpha_X(5) = 0.2;
    alpha_X(6) = 0.1;
    alpha_Y(1) = 0.04;
    alpha_Y(2) = 0.7;
    alpha_Y(3) = -0.1;
    alpha_Y(4) = -0.3;
    alpha_Y(5) = 0.1;
    alpha_Y(6) = 0.1;
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_t, eta_t, delta, 0., 0, 13435); //generate skewed-normal random errors
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_s, eta_s, delta, 0., 0, 53435);

    /************************************************************ Using the Bartlett kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the Daniell kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_daniell_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_daniell_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_daniell_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the QS kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_QS_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_QS_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_QS_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    /***************************************** Finish the Simulations for the Proposed Test: The Univariate Case ************************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment

	#if 0 //begin comment
	/**************************************************** Start the Simulations for El Himdi & Roy's test: The Bivariate Case ***********************************/
    /************************************************************************************************************************************************************/
    int T = 100;
	int number_sampl = 500;
    Matrix alpha(3,1), beta(3,1), lambda(3,1), sigma(2,1);
    alpha(1) = 0.01;
    alpha(2) = 0.04;
    alpha(3) = 0.02;
    beta(1) = 0.7;
    beta(2) = 0.4;
    beta(3) = -0.5;
    lambda(1) = 0.;
    lambda(2) = -1;
    lambda(3) = -0.8;
	sigma(1) = 1.5; // cf. Dgp::gen_CAR1 in dgp.h
	sigma(2) = 1.5; // cf. Dgp::gen_CAR1 in dgp.h
	//double rho = 0.2, threshold = 1.54; //set threshold equal to 1.54 for zero correlation
	double rho = 0.2, corr = 0.8; //set the correlation betwen X and Y1 to 0.8
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    Dep_tests dep_obj;
    Matrix cv(2, 1), asymp_cv(2, 1), empir_REJF(2, 1), asymp_REJF(2, 1);
    int lag_smooth = 5;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

	//============================================================================================================================================================
	lag_smooth = 10;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //============================================================================================================================================================
    lag_smooth = 20;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //============================================================================================================================================================
    lag_smooth = 25;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    /***************************************** Finish the Simulations for El Himdi & Roy's test: The Bivariate Case *********************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment

	/*Matrix alpha_x(2,1), alpha_y(2,1), beta_x(2,1), beta_y(2,1);
	alpha_x(1) = 0.001;
	alpha_x(2) = -0.002;
	beta_x(1) = 0.4;
	beta_x(2) = 0.1;
	alpha_y(1) = -0.001;
	alpha_y(2) = 0.002;
	beta_y(1) = 0.9;
	beta_y(2) = 0.78;
    Dgp dgp_obj (alpha_x, alpha_y, beta_x, beta_y);
    //Dgp dgp_obj;
    //dgp_obj.gen_MixedAR (X, Y, alpha, beta, lambda, rho, 543590823);
    Matrix X(50,2), Y(50,2);
    dgp_obj.gen_AR1 (X, Y, 0., 55235);*/
    /*int i = 1, TL = 10;
    while (i <= TL) {
    	x1(i,1) = X(TL-i+1,1);
    	x1(i,2) = X(TL-i+1,2);
    	x2(i,1) = X(TL-i+4,1);
    	x2(i,2) = X(TL-i+4,2);
    	++i;
	}
	x0(1,1) = 0.5;
	x0(1,2) = 0.01;*/
	//NReg nreg_obj;
	//NReg nreg_obj;
	//int lag = -2;
	//int lag_smooth = 10;
	/*int number_sampl = 500;
	Matrix sigma(2,1);
	sigma(1) = sqrt(2);
	sigma(2) = sqrt(0.5 + pow(rho, 2.) * 1.5);
	string output_filename, power_filename;
    power_filename =  "test_power_statistics.txt";
    ofstream pwr_out;
    pwr_out.open (power_filename.c_str(), ios::out);
    Power obj_power;
    double reject_rate = 0.;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, 0.95, 645345, pwr_out);
	cout << reject_rate << endl;*/
	//cout << "rho = " << rho <<": The value of the test statistic is " << dist_corr_obj.do_Test <bartlett_kernel> (TL, lag_smooth, alpha, beta, lambda, sigma)<< endl;
    //cout << "the denominator = " << nreg_obj.weight <epanechnikov_kernel> (X, x1, TL, 0.03) << endl;
    //cout << "the distance covariance test = " << nreg_obj.do_Test <bartlett_kernel,epanechnikov_kernel> (X, Y, 10, 10, 0.03) << endl;
    //cout << "the regression function = " << nreg_obj.var_Ux_ts <triangle_kernel> (X, x1, x2, x0, x0, TL, 0.03) << endl;
    //please do not comment out the lines below.
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    time = std::chrono::high_resolution_clock::now();
    //output << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //cout << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //output << "This program took " << std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count() << " seconds to run.\n";
    cout << "This program took " << std::chrono::duration_cast <std::chrono::milliseconds> (time-timelast).count() << " milliseconds to run.\n";
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    #if defined(_WIN64) || defined(_WIN32)
    system("PAUSE");
    #endif
    return 0;
}







