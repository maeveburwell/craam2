# CRAAM: Robust And Approximate Markov decision processes #

Craam is a **header-only** C++ library for solving Markov decision processes with support for handling uncertainty in transition probabilities. The library can handle uncertainties using both *robust*, or *optimistic* objectives.

The library includes Python and R interfaces. See below for detailed installation instructions.

When using the *robust objective*, adversarial nature chooses the worst plausible realization of the uncertain values. When using the *optimistic objective*, collaborative nature chooses the best plausible realization of the uncertain values. 

The library also provides tools for *basic simulation*, for constructing MDPs from *sample*s, and *value function approximation*. Objective functions supported are infinite horizon discounted MDPs, finite horizon MDPs, and stochastic shortest path \[Puterman2005\]. Some basic stochastic shortest path methods are also supported. The library assumes *maximization* over actions. The number of states and actions must be finite.

The library is based on two main data structures: MDP and MDPO. **MDP** is the standard model that consists of states 𝒮 and actions 𝒜. Note that robust solutions are constrained to be **absolutely continuous** with respect to *P*(*s*, *a*, ⋅). This is a hard requirement for all choices of ambiguity (or uncertainty).

The **MPDO** model adds a set of *outcomes* that model possible actions that can be taken by nature. Using outcomes makes it more convenient to capture correlations between the ambiguity in rewards and the uncertainty in transition probabilities. It also make it much easier to represent uncertainties that lie in small-dimensional vector spaces. Constraints for nature's distributions over outcomes are also supported.

The available algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution.


## Installing R Package ##

The R exposes most of the functions of the package. Method signatures are expected to change. The package should work on Linux, Mac, and Windows (with RTools 4.0+). R version 4.0 is required and the C++ compiler must support C+20 standard.

### Linux and Mac ###

A stable (and possibly stale) version of the package can be installed directly from the github repository using `remotes`:

``` R
install.packages("remotes")
remotes::install_github("marekpetrik/craam2","rcraam")
```
A development version can be installed from gitlab as follows:

``` R
install.packages("remotes")
remotes::install_gitlab("RLsquared/craam2","rcraam")
```

To use methods that use on Gurobi, you must download Gurobi (and get a license) and set `GUROBI_PATH` to the installations directory that has subdirectories `include` and `lib`.

To download and install a local development version, run:
``` bash
gitlab clone git@gitlab.com/RLSquared/craam2
cd craam2/rcraam
R CMD INSTALL . --preclean
```

### Windows ###

You also need to install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) 4.0 or later. If you want to avoid having to configure the compilation paths too, install [pkgbuild](https://cran.r-project.org/web/packages/pkgbuild/index.html). The code for that is:

``` R
install.packages(c("remotes","pkgbuild"))
remotes::install_github("marekpetrik/craam2","rcraam")
```

### Development ###

The C++ sources in directories `craam` and `includes` are currently replicated in `rcraam/inst/includes`. We are not using symlinks because they are not supported on Windows which makes it impossible to use `remotes::install_...`. The file `rcraam/copy_libs.sh` copies (running bash or similar) the latest version of the appropriate C++ files to `rcraam/inst/includes`.

## Installing C++ Library ##

It is sufficient to copy the entire root directory to a convenient location.

Numerous asserts are enabled in the code by default. To disable them, make sure to insert the following line *before* including any files:

``` cpp
#define NDEBUG
```
Or use the -DNDEBUG compiler switch.

To make sure that asserts are disabled, you may also want to double check the file `/craam/config.hpp` which is auto-generated by `cmake`.

The library has minimal dependencies and was tested on Linux. It also compiles on macOS using recent Xcode versions. It has not been tested on Windows.

### Requirements

-   At least C++17 compatible compiler, tested with C++20 compatible compiler (GCC 8+):

#### Optional Dependencies

-   [CMake](http://cmake.org/): 3.17.3 to build tests, command line executable, and the documentation
-   [Gurobi 9](http://gurobi.com) for using robust objectives that require a linear program solver. Set `GUROBI_PATH` to the location of the gurobi files (with subdirectories `include` and `lib`).
-   [OpenMP](http://openmp.org) to enable parallel computation
-   [Doxygen](http://doxygen.org%3E) 1.8.0+ to generate documentation
-   [Boost](http://boost.org) for compiling and running unit tests (`boost-devel` package, `libboost-all-dev` package on some distributions)


### Documentation

The project uses [Doxygen](http://www.stack.nl/~dimitri/doxygen/) for the documentation. To generate the documentation after generating the files, run:

``` bash
    $ cmake --build . --target docs
```

This automatically generates both HTML and PDF documentation in the folder `out`.

### Run unit tests

Note that Boost must be present in order to build the tests in the first place.

``` bash
    $ cmake .
    $ cmake --build . --target testit
```

## Build and Run Command-line Executable ##

To run a benchmark problem, download and decompress one of the following test files:

-   Small problem with 100 states: <https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip>
-   Medium problem with 2000 states (7zip): <https://www.dropbox.com/s/k0znc23xf9mpe5i/ms.7z>

These two benchmark problems were generated from a uniform random distribution.

Download the code.

``` bash
    $ git clone --depth 1 https://gitlab.com/RLsquared/craam2
```

Optionally, you can (re)install Eigen in the includes directory (requires bash or Cygwin on Windows). This is not necessary since the correct Eigen distribution is already included in the project git repository.

``` bash
    $ ./install_eigen.sh
```
To install it manually, download the latest version from <http://eigen.tuxfamily.org/> and install it under `include/eigen3`. A file `include/eigen3/Eigen/Core` should exist. 

We can now build the project as follows:
``` bash
    $ cmake -DCMAKE_BUILD_TYPE=Release .
    $ cmake --build . --target craam-cli
```

Finally, download and solve a simple benchmark problem:

``` bash
    $ mkdir data
    $ cd data
    $ wget https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip
    $ unzip ss.zip
    $ cd ..
    $ bin/craam-cli -i data/smallsize_test.csv -o data/smallsize_policy.csv
```

To see the list of command-line options, run:

``` bash
    $ bin/craam-cli -h
```

## C++ Development ##

The instructions above generate a release version of the project. The release version is optimized for speed, but lacks debugging symbols and many intermediate checks are eliminated. For development purposes, is better to use the Debug version of the code. This can be generated as follows:

``` bash
    $ cmake -DCMAKE_BUILD_TYPE=Debug .
    $ cmake --build .
```
The release version that omits many of the time-consuming debugging checks can be compiled as:

``` bash
    $ cmake -DCMAKE_BUILD_TYPE=Release .
    $ cmake --build .
```

By default, the project assumes that the [Gurobi](http://www.gurobi.com/) LP solver is available.  It is possible to disable the code that requires gurobi by uncommenting the fillowing line in CMakeLists.txt: 

```
set (GUROBI_USE FALSE)
```

CMake assumes by default that C++17 is available. If it is not, change the corresponding line in `CMakeLists.txt`.  If the necessary Gurobi files (see above) are not found in the expected directories, Gurobi support is disabled by CMake.

[QT creator](https://www.qt.io/download) is a nice IDE that can automatically parse and run cmake projects directly. As an alternative, CMake can be used to generate a [CodeBlocks](http://www.codeblocks.org/) project files too:


To help with development, CMake can be used to generate a [CodeBlocks](http://www.codeblocks.org/) project files too:

``` bash
  $ cmake . -G "CodeBlocks - Ninja"
```

To list other types of projects that CMake can generate, call:

``` bash
  $ cmake . -G
```

## Next Steps ##

### C++ Library ###


See the [online documentation](http://cs.unh.edu/~mpetrik/code/craam) or generate it locally as described above.

Unit tests provide some examples of how to use the library. For simple end-to-end examples, see `tests/benchmark.cpp` and `test/dev.cpp`. Targets `BENCH` and `DEV` build them respectively.

The main models supported are:

-   `craam::MDP` : plain MDP with no specific definition of ambiguity (can be used to compute robust solutions anyway)
-   `craam::RMDP` : an augmented model that adds nature's actions (so-called outcomes) to the model for convenience
-   `craam::impl::MDPIR` : an MDP with implementability constraints. See \[Petrik2016\].

The regular value-function based methods are in the header `algorithms/values.hpp` and the robust versions are in in the header `algorithms/robust_values.hpp`. There are 4 main value-function based methods:

-   `solve_vi`: Gauss-Seidel value iteration; runs in a single thread. -`solve_mpi`: Jacobi modified policy iteration; parallelized with OpenMP. Generally, modified policy iteration is vastly more efficient than value iteration.
-   `rsolve_vi`: Like the value iteration above, but also supports robust, risk-averse, or optimistic objectives.
-   `rsolve_mpi`: Like the modified policy iteration above, but it also supports robust, risk-averse, optimistic objective.

These methods can be applied to either an MDP or an RMDP.

The header `algorithms/occupancies.hpp` provides tools for converting the MDP to a transition matrix and computing the occupancy frequencies.

There are tools for building simulators and sampling from simulations in the header `Simulation.hpp` and methods for handling samples in `Samples.hpp`.

## References ##

-   \[Filar1997\] Filar, J., & Vrieze, K. (1997). Competitive Markov decision processes. Springer.
-   \[Puterman2005\] Puterman, M. L. (2005). Markov decision processes: Discrete stochastic dynamic programming. Handbooks in operations research and management …. John Wiley & Sons, Inc.
-   \[Iyengar2005\] Iyengar, G. N. G. (2005). Robust dynamic programming. Mathematics of Operations Research, 30(2), 1–29.
-   \[Petrik2014\] Petrik, M., Subramanian S. (2014). RAAM : The benefits of robustness in approximating aggregated MDPs in reinforcement learning. In Neural Information Processing Systems (NIPS).
-   \[Petrik2016\] Petrik, M., & Luss, R. (2016). Interpretable Policies for Dynamic Product Recommendations. In Uncertainty in Artificial Intelligence (UAI).
