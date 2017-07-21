CRAAM: Robust And Approximate Markov decision processes
================

[![Build Status](https://travis-ci.org/marekpetrik/CRAAM.svg?branch=master)](https://travis-ci.org/marekpetrik/CRAAM)

Craam is a **header-only** C++ library for solving Markov decision processes with *regular*, *robust*, or *optimistic* objectives. The optimistic obective is the opposite of robust, in which nature chooses the best possible realization of the uncertain values. The library also provides tools for *basic simulation*, for constructing MDPs from *sample*s, and *value function approximation*. Objective functions supported are infinite horizon discounted MDPs, finite horizon MDPs, and stochastic shortest path \[Puterman2005\]. Some basic stochastic shortest path methods are also supported. The library assumes *maximization* over actions. The number of states and actions must be finite.

The library is build around two main data structures: MDP and RMDP. **MDP** is the standard model that consists of states 𝒮 and actions 𝒜. The robust solution for an MDP would satisfy, for example, the following Bellman optimality equation:
*v*(*s*)=max<sub>*a* ∈ 𝒜</sub>min<sub>*p* ∈ *Δ*</sub>{∑<sub>*s*′∈𝒮</sub>*p*(*s*′)(*r*(*s*,*a*,*s*′)+*γ*  *v*(*s*′)) : ∥*p*−*P*(*s*,*a*,⋅)∥≤*ψ*, *p*≪*P*(*s*,*a*,⋅)} .
 Note that *p* is constrained to be **absolutely continuous** with respect to *P*(*s*, *a*, ⋅). This is a hard requirement for all choices of ambiguity (or uncertainty).

The **RMPD** model adds a set of *outcomes* that model possible actions that can be taken by nature. In that case, the robust solution may for example satisfy the following Bellman optimality equation:
*v*(*s*)=max<sub>*a* ∈ 𝒜</sub>min<sub>*o* ∈ 𝒪</sub>∑<sub>*s*′∈𝒮</sub>*P*(*s*, *a*, *o*, *s*′)(*r*(*s*, *a*, *o*, *s*′) + *γ* *v*(*s*′)) .
 Using outcomes makes it more convenient to capture correlations between the ambiguity in rewards and the uncertainty in transition probabilities. It also make it much easier to represent uncertainties that lie in small-dimensional vector spaces. The equation above uses the worst outcome, but in general distributions over outcomes are supported.

The available algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution.

A python interface is also supported. See the instructions below.

Installing C++ Library
======================

No installation is required. Numerous asserts are enabled in the code by default. To disable them, insert the following line *before* importing any files:

``` cpp
#define NDEBUG
```

The library has minimal dependencies and was tested on Linux and MacOS operating systems. It has not been tested on Windows.

### Requirements

-   C++14 compatible compiler:
    -   Tested with Linux GCC 4.9.2,5.2.0,6.1.0; does not work with GCC 4.7, 4.8.
    -   Tested with Linux Clang 3.6.2 (and maybe 3.2+).
-   [Eigen](http://eigen.tuxfamily.org) 3+ for computing occupancy frequencies

#### Optional Dependencies

-   [CMake](http://cmake.org/): 3.1.0 to build tests and documentation
-   [OpenMP](http://openmp.org) to enable parallel computation
-   [Doxygen](http://doxygen.org%3E) 1.8.0+ to generate documentation
-   [Boost](http://boost.org) for compiling and running unit tests

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

### Build a benchmark executable

To run a benchmark problem, download and decompress one of the following test files:

-   Small problem with 100 states: <https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip>
-   Medium problem with 2000 states (7zip): <https://www.dropbox.com/s/k0znc23xf9mpe5i/ms.7z>

These two benchmark problems were generated randomly.

The small benchmark example, for example, can be executed as follows:

``` bash
    $ cmake --build . --target benchmark
    $ mkdir data
    $ cd data
    $ wget https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip
    $ unzip ss.zip
    $ cd ..
    $ bin/benchmark data/smallsize_test.csv
```

Install Python Interface
========================

This install a package `craam`, with most of the classes and method provided by `craam.crobust`.

Requirements
------------

-   Python 3.5+ (Python 2 is NOT supported)
-   Setuptools 7.0+
-   Numpy 1.8+
-   Cython 0.24+

Installation
------------

To install the Python extension, first compile the C++ library as described above. Then go to the `python` subdirectory and run:

``` bash
  $ python3 setup.py install --user 
```

Omit `--user` to install the package for all users rather than just the current one.

Development
===========

The instruction above generate a release version of the project. The release version is optimized for speed, but lacks debugging symbols and many intermediate checks are eliminated. For development purposes, is better to use the Debug version of the code. This can be generated as follows:

``` bash
    $ cmake -DCMAKE_BUILD_TYPE=Debug .
    $ cmake --build .
```

To help with development, Cmake can be used to generate a [CodeBlocks](http://www.codeblocks.org/) project files too:

``` bash
  $ cmake . -G "CodeBlocks - Ninja"
```

To list other types of projects that Cmake can generate, call:

``` bash
  $ cmake . -G
```

Installing Python Package
=========================

A convenient way to develop Python packages is to install them in the development mode as:

``` bash
  $ python3 setup.py develop --user 
```

In the development mode, the python files are not copied on installation, but rather their development version is used. This means that it is not necessary to reinstall the package to reflect code changes. **Cython note**: Any changes to the cython code require that the package is rebuilt and reinstalled.

Next Steps
==========

C++ Library
-----------

See the [online documentation](http://cs.unh.edu/~mpetrik/code/craam) or generate it locally as described above.

Unit tests provide some examples of how to use the library. For simple end-to-end examples, see `tests/benchmark.cpp` and `test/dev.cpp`. Targets `BENCH` and `DEV` build them respectively.

The main models supported are: - `craam::MDP` : plain MDP with no definition of uncertainty - `craam::RMDP` : a robust/uncertain with discrete outcomes with L1 constraints on the uncertainty - `craam::impl::MDPIR` : an MDP with implementatbility constraints. See \[Petrik2016\].

The regular value-function based methods are in the header `algorithms/values.hpp` and the robust versions are in in the header `algorithms/robust_values.hpp`. There are 4 main value-function based methods:

<table style="width:85%;">
<colgroup>
<col width="33%" />
<col width="51%" />
</colgroup>
<thead>
<tr class="header">
<th>Method</th>
<th>Algorithm</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><code>solve_vi</code></td>
<td>Gauss-Seidel value iteration; runs in a single thread.</td>
</tr>
<tr class="even">
<td><code>solve_mpi</code></td>
<td>Jacobi modified policy iteration; parallelized with OpenMP. Generally, modified policy iteration is vastly more efficient than value iteration.</td>
</tr>
<tr class="odd">
<td><code>rsolve_vi</code></td>
<td>Like the value iteration above, but also supports robust, risk-averse, or optimistic objectives.</td>
</tr>
<tr class="even">
<td><code>rsolve_mpi</code></td>
<td>Like the modified policy iteration above, but it also supports robust, risk-averse, optimistic objective.</td>
</tr>
</tbody>
</table>

These methods can be applied to eithen an MDP or an RMDP.

The header `algorithms/occupancies.hpp` provides tools for converting the MDP to a transition matrix and computing the occupancy frequencies.

There are tools for building simulators and sampling from simulations in the header `Simulation.hpp` and methods for handling samples in `Samples.hpp`.

Python Interface
----------------

The python interface closely mirrors the C++ classes. The following main types of plain and robust MDPs supported:

-   `craam.MDP` : plain MDP with no definition of uncertainty
-   `craam.RMDP` : a robust/uncertain with discrete outcomes with L1 constraints on the uncertainty
-   `craam.MDPIR` : an MDP with implementatbility constraints. See \[Petrik2016\].

The classes support the following main optimization algorithms:

<table style="width:85%;">
<colgroup>
<col width="33%" />
<col width="51%" />
</colgroup>
<thead>
<tr class="header">
<th>Method</th>
<th>Algorithm</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>solve_vi</td>
<td>Gauss-Seidel value iteration; runs in a single thread.</td>
</tr>
<tr class="even">
<td>solve_mpi</td>
<td>Jacobi modified policy iteration; parallelized with OpenMP. Generally, modified policy iteration is vastly more efficient than value iteration.</td>
</tr>
<tr class="odd">
<td>rsolve_vi</td>
<td>Like the value iteration above, but also supports robust, risk-averse, or optimistic objectives.</td>
</tr>
<tr class="even">
<td>rsolve_mpi</td>
<td>Like the modified policy iteration above, but it also supports robust, risk-averse, optimistic objective.</td>
</tr>
</tbody>
</table>

States, actions, and outcomes (actions of nature) are represented by 0-based contiguous indexes. The actions are indexed independently for each state and the outcomes are indexed independently for each state and action pair.

Transitions are added through function add\_transition. New states, actions, or outcomes are automatically added based on the new transition.

Other classes are available to support simulating MDPs and constructing them from samples:

-   `craam.crobust.SimulatorMDP` : Simulates an MDP for a given deterministic or randomized policy
-   `craam.crobust.DiscreteSamples` : Collection of state to state transitions as well as samples of initial states. All states and actions are identified by integers.
-   `craam.crobust.SampledMDP` : Constructs an MDP from samples in `DiscreteSamples`.

### Solving a Simple MDP

The following code solves a simple MDP problem precisely using modified policy iteration.

``` python
from craam import crobust
import numpy as np

states = 100
P1 = np.random.rand(states,states)
P1 = np.diag(1/np.sum(P1,1)).dot(P1)
P2 = np.random.rand(states,states)
P2 = np.diag(1/np.sum(P2,1)).dot(P2)
r1 = np.random.rand(states)
r2 = np.random.rand(states)

transitions = np.dstack((P1,P2))
rewards = np.column_stack((r1,r2))

mdp = crobust.MDP(states,0.99)
mdp.from_matrices(transitions,rewards)
value,policy,residual,iterations = mdp.solve_mpi(100)

print('Value function s0-s9:', value[:10])
```

    ## Value function s0-s9: [ 68.59164877  68.17255088  69.01759101  68.50824648  68.77114289
    ##   68.84949497  68.86934182  69.00629372  68.59825972  69.0319304 ]

This example can be easily converted to a robust MDP by appropriately defining additional outcomes (the options available to nature) with transition matrices and rewards.

References
----------

-   \[Filar1997\] Filar, J., & Vrieze, K. (1997). Competitive Markov decision processes. Springer.
-   \[Puterman2005\] Puterman, M. L. (2005). Markov decision processes: Discrete stochastic dynamic programming. Handbooks in operations research and management …. John Wiley & Sons, Inc.
-   \[Iyengar2005\] Iyengar, G. N. G. (2005). Robust dynamic programming. Mathematics of Operations Research, 30(2), 1–29.
-   \[Petrik2014\] Petrik, M., Subramanian S. (2014). RAAM : The benefits of robustness in approximating aggregated MDPs in reinforcement learning. In Neural Information Processing Systems (NIPS).
-   \[Petrik2016\] Petrik, M., & Luss, R. (2016). Interpretable Policies for Dynamic Product Recommendations. In Uncertainty in Artificial Intelligence (UAI).
