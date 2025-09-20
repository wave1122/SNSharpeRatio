# About The Project

This repository provides C++ codes used to conduct Monte Carlo simulations in the paper: `Inference with Sharp Ratios via Self-Normalization`

# Pre-requisites

* [GNU C++ compiler (GNU GCC 13.3.0)](https://gcc.gnu.org/)

* [GSL - GNU Scientific Library 2.8](https://www.gnu.org/software/gsl/)

* [Boost C++ library 1.78.0](https://www.boost.org/)

* [Dlib 20.0](https://dlib.net/)

* [Ubuntu 24.04.3 LTS](https://ubuntu.com/)

  

# Build

On an Ubuntu machine, a binary executable can be built from a `cpp file` by running following shell script from the terminal:

```sh
g++ -Wno-deprecated -O3 -Wfloat-equal -Wfatal-errors -m64 -std=gnu++17 -fopenmp -ldlib -lX11 -lpthread -lboost_thread -Wunknown-pragmas -Wall -Waggressive-loop-optimizations -mavx2 -march=native -mtune=native -I/<path-to-the-folder-containing-source-codes>/ -I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/4.9.3/include -I/<path-to-gsl-2.8>/include -I/usr/local/include -I/<path-to-dlib-20.0-library>/include -c/<path-to-the-folder-containing-source-codes>/CPP/<main file with extension *.cpp> -o .objs/main.o

g++ -L/<path-to-gsl-2.8>/lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib -L/<path-to-dlib-20.0-library>/lib -o <name-of-the-binary-to-be-built> .objs/main.o  -fopenmp -O3 -m64 -lgsl -lgslcblas -lm -fopenmp -lpthread -lboost_thread -lX11 -ldlib -lblas -llapack  -lgsl -lgslcblas -lm
```

# List of C++ main files

| C++ main file                                                | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `rejection_rates_sharpe_gaussian_v2.cpp`                     | to calculate size and power of the tests for the equality between two Sharpe ratios when random samples are drawn from a bivariate Gaussian distribution |
| `rejection_rates_sharpe_tdist_v2.cpp`                        | to calculate size and power of the tests for the equality between two Sharpe ratios when random samples are drawn from a bivariate Student's *t* distribution |
| `rejection_rates_sharpe_garch_gaussian_v2.cpp`               | to calculate size and power of the tests for the equality between two Sharpe ratios when random samples are drawn from a bivariate GARCH(1,1) process with Gaussian innovation |
| `rejection_rates_sharpe_garch_tdist_v2.cpp`                  | to calculate size and power of the tests for the equality between two Sharpe ratios when random samples are drawn from a bivariate GARCH(1,1) process with Student's *t* innovation |
| `rejection_rates_sharpe_argarch_gaussian_v2.cpp`             | to calculate size and power of the tests for the equality between two Sharpe ratios when random samples are drawn from a bivariate AR(1)-GARCH(1,1) process with Gaussian innovation |
| `rejection_rates_sharpe_argarch_tdist_v2.cpp`                | to calculate size and power of the tests for the equality between two Sharpe ratios when random samples are drawn from a bivariate AR(1)-GARCH(1,1) process with Student *t* innovation |
| `rejection_rates_max_sq_sharpe_gaussian_v2.cpp`              | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of traded factors when random samples are drawn from a multivariate Gaussian distribution |
| `rejection_rates_max_sq_sharpe_tdist_v2.cpp`                 | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of traded factors when random samples are drawn from a multivariate Student's *t* distribution |
| `rejection_rates_max_sq_sharpe_garch_gaussian_v2.cpp`        | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of traded factors when random samples are drawn from a multivariate GARCH(1,1) process with Gaussian innovation |
| `rejection_rates_max_sq_sharpe_garch_tdist_v2.cpp`           | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of traded factors when random samples are drawn from a multivariate GARCH(1,1) process with Student's *t* innovation |
| `rejection_rates_max_sq_sharpe_argarch_gaussian_v2.cpp`      | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of traded factors when random samples are drawn from a multivariate AR(1)-GARCH(1,1) process with Gaussian innovation |
| `rejection_rates_max_sq_sharpe_argarch_tdist_v2.cpp`         | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of traded factors when random samples are drawn from a multivariate AR(1)-GARCH(1,1) process with Student's *t* innovation |
| `rejection_rates_max_sq_sharpe_mimicking_gaussian_v2.cpp`    | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of non-traded factors when random samples are drawn from a multivariate Gaussian distribution |
| `rejection_rates_max_sq_sharpe_mimicking_tdist_v2.cpp`       | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of non-traded factors when random samples are drawn from a multivariate Student's *t* distribution |
| `rejection_rates_max_sq_sharpe_mimicking_garch_gaussian_v2.cpp` | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of non-traded factors when random samples are drawn from a multivariate GARCH(1,1) process with Gaussian innovation |
| `rejection_rates_max_sq_sharpe_mimicking_garch_tdist_v2.cpp` | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of non-traded factors when random samples are drawn from a multivariate GARCH(1,1) process with Student's *t* innovation |
| `rejection_rates_max_sq_sharpe_mimicking_argarch_gaussian_v2.cpp` | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of non-traded factors when random samples are drawn from a multivariate AR(1)-GARCH(1,1) process with Gaussian innovation |
| `rejection_rates_max_sq_sharpe_mimicking_argarch_tdist_v2.cpp` | to calculate size and power of the tests for the equality between the maximum squared Sharpe ratios of  two non-nested sets of non-traded factors when random samples are drawn from a multivariate AR(1)-GARCH(1,1) process with Student's *t* innovation |

All the graphs reported in the paper are generated by the Jupyter Notebook: `plots_v2.ipynb`.

# License

Distributed under the MIT License. See `LICENSE.txt` for more information.

# Contact

Ba Chu -  ba.chu@carleton.ca

Project Link: [https://github.com/wave1122/DcorrTest](https://github.com/wave1122/DcorrTest)
