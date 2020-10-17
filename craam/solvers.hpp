﻿// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "craam/MDP.hpp"
#include "craam/MDPO.hpp"
#include "craam/Solution.hpp"
#include "craam/algorithms/bellman_mdp.hpp"
#include "craam/algorithms/bellman_mdpo.hpp"
#include "craam/algorithms/iteration_methods.hpp"
#include "craam/algorithms/linprog.hpp"
#include "craam/algorithms/nature_declarations.hpp"
#include "craam/modeltools.hpp"
#include "craam/optimization/gurobi.hpp"

#include <cmath>

namespace craam {

/**
 * @defgroup ValueIteration
 *
 * Gauss-Seidel variant of value iteration (not parallelized).
 *
 * This function is suitable for computing the value function of a finite state
 * MDP. If the states are ordered correctly, one iteration is enough to compute the
 * optimal value function. Since the value function is updated from the last state
 * to the first one, the states should be ordered in the temporal order.
 * @param mdp The MDP to solve
 * @param discount Discount factor.
 * @param valuefunction Initial value function. Passed by value, because it is
 * modified. Optional, use all zeros when not provided. Ignored when size is 0.
 * @param policy Partial policy specification. Optimize only actions that are policy[state] = -1
 * @param rpolicy Randomized policy with a probability for each state and action
 * @param iterations Maximal number of iterations to run
 * @param maxresidual Stop when the maximal residual falls below this value.
 * @param progress An optional function for reporting progress and can
                return false to stop computation
 *
 * @returns Solution that can be used to compute the total return, or the optimal
policy.
 */

/**
 * @defgroup ModifiedPolicyIteration
 *
 * Modified policy iteration using Jacobi value iteration in the inner loop.
 * This method generalizes modified policy iteration to robust MDPs.
 * In the value iteration step, both the action *and* the outcome are fixed.
 *
 * Note that the total number of iterations will be bounded by iterations_pi *
 * iterations_vi
 * @param discount Discount factor
 * @param valuefunction Initial value function
 * @param policy Partial policy specification. Optimize only actions that are
 * policy[state] = -1
 * @param rpolicy Randomized policy with a probability for each state and action
 * @param iterations_pi Maximal number of policy iteration steps
 * @param maxresidual_pi Stop the outer policy iteration when the residual drops
 * below this threshold.
 * @param iterations_vi Maximal number of inner loop value iterations
 * @param maxresidual_vi Stop policy evaluation when the policy residual drops
 * below maxresidual_vi * last_policy_residual
 * @param progress An optional function for reporting progress and can
                return false to stop computation
 *
 * @return Computed (approximate) solution
 */

/**
 * @defgroup PolicyIteration
 *
 * Policy iteration using parallel action updated and Eigen to compute matrix
 * inverse. Since this method is based on dense matrices, it does not scale
 * particularly well.
 *
 * The method stop when the residual reaches the specified threshold or the policy
 * no longer changes.
 *
 * This method generalizes modified policy iteration to robust MDPs.
 * In the value iteration step, both the action *and* the outcome are fixed.
 *
 * Note that the total number of iterations will be bounded by iterations_pi *
 * iterations_vi
 * @param discount Discount factor
 * @param valuefunction Initial value function
 * @param policy Partial policy specification. Optimize only actions that are
 * policy[state] = -1
 * @param rpolicy Randomized policy with a probability for each state and action
 * @param iterations_pi Maximal number of policy iteration steps
 * @param maxresidual_pi Stop the outer policy iteration when the residual drops
 * below this threshold.
 * @param progress An optional function for reporting progress and can
                return false to stop computation
 *
 * @return Computed (approximate) solution
 */

// **************************************************************************
// Plain MDP methods
// **************************************************************************

/**
 * @ingroup ValueIteration
 */
inline DetermSolution
solve_vi(const MDP& mdp, prec_t discount, numvec valuefunction = numvec(0),
         const indvec& policy = indvec(0), unsigned long iterations = MAXITER,
         prec_t maxresidual = SOLPREC,
         const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::vi_gs(algorithms::PlainBellman(mdp, policy), discount,
                             move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup ModifiedPolicyIteration
 */
inline DetermSolution
solve_mpi(const MDP& mdp, prec_t discount, const numvec& valuefunction = numvec(0),
          const indvec& policy = indvec(0), unsigned long iterations_pi = MAXITER,
          prec_t maxresidual_pi = SOLPREC, unsigned long iterations_vi = MAXITER,
          prec_t maxresidual_vi = 0.9,
          const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::mpi_jac(algorithms::PlainBellman(mdp, policy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, progress);
}

/**
 * Computes occupancy frequencies using matrix representation of transition
 * probabilities. This method requires computing a matrix inverse.
 *
 * @param init Initial distribution (alpha)
 * @param discount Discount factor (gamma)
 * @param policies The policy (indvec) or a pair of the policy and the policy
 *        of nature (pair<indvec,vector<numvec> >). The nature is typicall
 *        a randomized policy
 */
inline numvec occupancies(const MDP& mdp, const Transition& initial, prec_t discount,
                          const indvec& policy) {

    check_model(mdp);
    return algorithms::occfreq_mat(algorithms::PlainBellman(mdp), initial, discount,
                                   policy);
}

/**
 * @ingroup PolicyIteration
 */
inline DetermSolution
solve_pi(const MDP& mdp, prec_t discount, numvec valuefunction = numvec(0),
         const indvec& policy = indvec(0), unsigned long iterations = MAXITER,
         prec_t maxresidual = SOLPREC,
         const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::pi(algorithms::PlainBellman(mdp, policy), discount,
                          move(valuefunction), iterations, maxresidual, progress);
}

#ifdef GUROBI_USE
/**
 * Solves the MDP using the primal formulation (using value functions)
 *
 * Solves the linear program
 * min_v  1^T v
 * s.t.   P_a v >= r_a  for all a
 *
 * @param mdp Markov decision process
 * @param discount Discount factor
 *
 * @return Solution
 */
inline DetermSolution solve_lp(const MDP& mdp, prec_t discount,
                               const indvec& policy = indvec(0),
                               GRBEnv& env = *get_gurobi()) {

    // TODO add support for this, the parameter is here only for
    // future signature compatibility
    if (policy.size() > 0) {
        throw logic_error("Partial policy specification is not supported by the linear "
                          "program MDP fomrulation yet.");
    }

    check_model(mdp);
    return algorithms::solve_lp_primal(env, mdp, discount);
}
#endif // GUROBI_USE

// **************************************************************************
// Plain MDP methods with a randomized policy (the optimal one is deterministic
// but a stochastic partial policy may be provided)
// **************************************************************************

/**
 * \ingroup ValueIteration
 */
inline RandSolution solve_vi_r(
    const MDP& mdp, prec_t discount, numvec valuefunction = numvec(0),
    const numvecvec& policy = numvecvec(0), unsigned long iterations = MAXITER,
    prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {

    check_model(mdp);
    if (!policy.empty() && policy.size() != mdp.size())
        throw invalid_argument("Policy length does not match number of states.");

    return algorithms::vi_gs(algorithms::PlainBellmanRand(mdp, policy), discount,
                             move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup ModifiedPolicyIteration
 */
inline RandSolution solve_mpi_r(
    const MDP& mdp, prec_t discount, const numvec& valuefunction = numvec(0),
    const numvecvec& policy = numvecvec(0), unsigned long iterations_pi = MAXITER,
    prec_t maxresidual_pi = SOLPREC, unsigned long iterations_vi = MAXITER,
    prec_t maxresidual_vi = 0.9,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {

    check_model(mdp);
    if (!policy.empty() && policy.size() != mdp.size())
        throw invalid_argument("Policy length does not match number of states.");

    return algorithms::mpi_jac(algorithms::PlainBellmanRand(mdp, policy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, progress);
}

/*
Computes occupancy frequencies using matrix representation of transition
probabilities. This method requires computing a matrix inverse.

\param init Initial distribution (alpha)
\param discount Discount factor (gamma)
\param policies The policy (indvec) or a pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically
        a randomized policy
inline numvec occupancies(const MDP& mdp, const Transition& initial, prec_t discount,
                          const indvec& policy) {

    return algorithms::occfreq_mat(algorithms::PlainBellman(mdp), initial, discount,
                                   policy);
}*/

/**
 * @ingroup PolicyIteration
 */
inline RandSolution solve_pi_r(
    const MDP& mdp, prec_t discount, numvec valuefunction = numvec(0),
    const numvecvec& policy = numvecvec(0), unsigned long iterations = MAXITER,
    prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {

    check_model(mdp);
    if (!policy.empty() && policy.size() != mdp.size())
        throw invalid_argument("Policy length does not match number of states.");

    return algorithms::pi(algorithms::PlainBellmanRand(mdp, policy), discount,
                          move(valuefunction), iterations, maxresidual, progress);
}

// **************************************************************************
// Robust SA-Rect MDP methods
// **************************************************************************

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s,a-rectangular nature.
 */
inline SARobustSolution
rsolve_vi(const MDP& mdp, prec_t discount, const algorithms::SANature& nature,
          numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
          const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::vi_gs(algorithms::SARobustBellman(mdp, nature, policy), discount,
                             move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s,a-rectangular nature.
 *
 * WARNING: There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 * The divergence of a similar algorithm is shown in:
 * Condon, A. (1993). On algorithms for simple stochastic games.
 * Advances in Computational Complexity Theory, DIMACS Series in Discrete Mathematics
 * and Theoretical Computer Science, 13, 51–71.
 *
 */
inline SARobustSolution rsolve_mpi(
    const MDP& mdp, prec_t discount, const algorithms::SANature&& nature,
    const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
    unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::mpi_jac(algorithms::SARobustBellman(mdp, nature, policy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, progress);
}

/**
 * @ingroup PolicyIteration
 * Robust policy iteration with an s,a-rectangular nature.
 *
 * WARNING: This method is not guaranteed to converge to the optimal solution
 * or converge at all. The divergence of the method is shown in:
 * Condon, A. (1993). On algorithms for simple stochastic games.
 * Advances in Computational Complexity Theory, DIMACS Series in Discrete Mathematics
 * and Theoretical Computer Science, 13, 51–71.
 */
inline SARobustSolution
rsolve_pi(const MDP& mdp, prec_t discount, const algorithms::SANature& nature,
          numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
          const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::pi(algorithms::SARobustBellman(mdp, nature, policy), discount,
                          move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup PartialPolicyIteration
 * Robust partial policy iteration with an sa-rectangular nature.
 *
 * Uses policy iteration to solve the inner MDP problem that corresponds
 * to the nature. This can be very efficient, but may not scale to large
 * problems.
 *
 * This method is guaranteed to converge to the optimal value function
 * and policy.
 */
inline SARobustSolution rsolve_ppi(
    const MDP& mdp, prec_t discount, const algorithms::SANature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::rppi(algorithms::SARobustBellman(mdp, nature, policy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::pi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 * Robust partial policy iteration with an sa-rectangular nature.
 *
 * Uses modified policy iteration to solve the inner MDP problem that corresponds
 * to the nature. This algorithm scales better than PPI with policy ieration
 * to problems with large state spaces.
 *
 * This method is guaranteed to converge to the optimal value function
 * and policy.
 */
inline SARobustSolution rsolve_mppi(
    const MDP& mdp, prec_t discount, const algorithms::SANature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::rppi(algorithms::SARobustBellman(mdp, nature, policy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            std::pow(discount, 2), algorithms::MDPSolver::mpi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 * Robust partial policy iteration with an sa-rectangular nature.
 *
 * Uses value iteration to solve the inner MDP problem that corresponds
 * to the nature. This algorithm scales better than PPI with policy ieration
 * to problems with large state spaces, but probably worse than MPI.
 *
 * This method is guaranteed to converge to the optimal value function
 * and policy.
 */
inline SARobustSolution rsolve_vppi(
    const MDP& mdp, prec_t discount, const algorithms::SANature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::rppi(algorithms::SARobustBellman(mdp, nature, policy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::vi, progress);
}

// **************************************************************************
// Robust S-Rect MDP methods
// **************************************************************************

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s-rectangular nature.
 */
inline SRobustSolution rsolve_s_vi(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    auto rpolicy = policy_det2rand(mdp, policy);
    return algorithms::vi_gs(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                             move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s-rectangular nature.
 */
inline SRobustSolution rsolve_s_vi_r(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const numvecvec& rpolicy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::vi_gs(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                             move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s-rectangular nature.
 *
 * WARNING: There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 */
inline SRobustSolution rsolve_s_mpi(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
    unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    auto rpolicy = policy_det2rand(mdp, policy);
    return algorithms::mpi_jac(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, progress);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s-rectangular nature.
 *
 * WARNING: The algorithm may loop forever without converging.
 * There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 *
 */
inline SRobustSolution rsolve_s_mpi_r(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    const numvec& valuefunction = numvec(0), const numvecvec& rpolicy = numvecvec(0),
    unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
    unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::mpi_jac(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                               valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, progress);
}

/**
 * @ingroup PolicyIteration
 * Robust policy iteration with an s-rectangular nature.
 *
 * WARNING: The algorithm may loop forever without converging.
 * There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 *
 */
inline SRobustSolution rsolve_s_pi(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    auto rpolicy = policy_det2rand(mdp, policy);
    return algorithms::pi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                          move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup PolicyIteration
 * Robust policy iteration with an s-rectangular nature.
 *
 * WARNING: The algorithm may loop forever without converging.
 * There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 *
 */
inline SRobustSolution rsolve_s_pi_r(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const numvecvec& rpolicy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::pi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                          move(valuefunction), iterations, maxresidual, progress);
}

/**
 * @ingroup PartialPolicyIteration
 *
 * Partial policy iteration. Computes approximations to the value function with
 * an increasing precision. Each decision maker's policy is evaluated using
 * an MDP solver.
 *
 * Uses policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 *
 */
inline SRobustSolution rsolve_s_ppi(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    auto rpolicy = policy_det2rand(mdp, policy);
    return algorithms::rppi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::pi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 *
 * Partial policy iteration. Computes approximations to the value function with
 * an increasing precision. Each decision maker's policy is evaluated using
 * an MDP solver.
 *
 * Uses modified policy iteration to solve the inner MDP problem that corresponds
 * to the nature. This scales better than the default policy iteration.
 *
 */
inline SRobustSolution rsolve_s_mppi(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    auto rpolicy = policy_det2rand(mdp, policy);
    return algorithms::rppi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::mpi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 *
 * Partial policy iteration. Computes approximations to the value function with
 * an increasing precision. Each decision maker's policy is evaluated using
 * an MDP solver.
 *
 * Uses value iteration to solve the inner MDP problem that corresponds
 * to the nature. This algorithm scales better than PPI with policy ieration
 * to problems with large state spaces, but probably worse than MPI.
 *
 */
inline SRobustSolution rsolve_s_vppi(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    auto rpolicy = policy_det2rand(mdp, policy);
    return algorithms::rppi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::vi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 *
 * Uses policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 *
 * @param mdp Definition of the MDP
 * @param discount Discount factor
 * @param valuefunction Initial value function
 * @param policy Partial policy specification. Optimize only actions that are
 * policy[state] = -1
 * @param rpolicy Randomized policy with a probability for each state and action
 * @param iterations_pi Maximal number of policy iteration steps
 * @param maxresidual_pi Stop the outer policy iteration when the residual drops
 * below this threshold.
 * @param progress An optional function for reporting progress and can
                return false to stop computation
 */
inline SRobustSolution rsolve_s_ppi_r(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const numvecvec& rpolicy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    return algorithms::rppi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::pi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 *
 * Uses modified policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 *
 * This policy uses randomized policies as inputs
 *
 * @param mdp Definition of the MDP
 * @param discount Discount factor
 * @param valuefunction Initial value function
 * @param policy Partial policy specification. Optimize only actions that are
 * policy[state] = -1
 * @param rpolicy Randomized policy with a probability for each state and action
 * @param iterations_pi Maximal number of policy iteration steps
 * @param maxresidual_pi Stop the outer policy iteration when the residual drops
 * below this threshold.
 * @param progress An optional function for reporting progress and can
                return false to stop computation

 */
inline SRobustSolution rsolve_s_mppi_r(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const numvecvec& rpolicy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    return algorithms::rppi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::mpi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 *
 * Uses modified policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 *
 * Uses value iteration to solve the inner MDP problem that corresponds
 * to the nature.This algorithm scales better than PPI with policy ieration
 * to problems with large state spaces, but probably worse than MPI.
 *
 * @param rpolicy Randomized policy with a probability for each state and action
 */
inline SRobustSolution rsolve_s_vppi_r(
    const MDP& mdp, prec_t discount, const algorithms::SNature& nature,
    numvec valuefunction = numvec(0), const numvecvec& rpolicy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    return algorithms::rppi(algorithms::SRobustBellman(mdp, nature, rpolicy), discount,
                            move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::vi, progress);
}

// **************************************************************************
// Plain MDPO methods
// **************************************************************************

/**
 * This method treats outcomes as another transition with uniform probabilities.
 * It does not compute the robust solution. The MDPO is thus trated simply
 * as an MDP.
 *
 * \ingroup ValueIteration
 */
inline DetermSolution
solve_vi(const MDPO& mdp, prec_t discount, numvec valuefunction = numvec(0),
         const indvec& policy = indvec(0), unsigned long iterations = MAXITER,
         prec_t maxresidual = SOLPREC,
         const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    auto solution = algorithms::vi_gs(
        algorithms::SARobustOutcomeBellman(mdp, algorithms::nats::average(), policy),
        discount, move(valuefunction), iterations, maxresidual, progress);

    // remove the nature's choice from the solution since there is no
    // nature's choice
    return DetermSolution(solution.valuefunction, unzip(solution.policy).first,
                          solution.residual, solution.iterations, solution.time,
                          solution.status);
}

/**
 * This method treats outcomes as another transition with uniform probabilities.
 * It does not compute the robust solution. The MDPO is thus trated simply
 * as an MDP.
 *
 * @ingroup ModifiedPolicyIteration
 */
inline DetermSolution
solve_mpi(const MDPO& mdp, prec_t discount, const numvec& valuefunction = numvec(0),
          const indvec& policy = indvec(0), unsigned long iterations_pi = MAXITER,
          prec_t maxresidual_pi = SOLPREC, unsigned long iterations_vi = MAXITER,
          prec_t maxresidual_vi = 0.9, bool print_progress = false,
          const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    auto solution = algorithms::mpi_jac(
        algorithms::SARobustOutcomeBellman(mdp, algorithms::nats::average(), policy),
        discount, valuefunction, iterations_pi, maxresidual_pi, iterations_vi,
        maxresidual_vi, progress);

    // remove the nature's choice from the solution since there is no
    // nature's choice
    return DetermSolution(solution.valuefunction, unzip(solution.policy).first,
                          solution.residual, solution.iterations, solution.time,
                          solution.status);
}

// **************************************************************************
// Robust SA-Rect MDPO methods
// **************************************************************************

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s,a-rectangular nature.
 */
inline SARobustSolution
rsolve_vi(const MDPO& mdp, prec_t discount, const algorithms::SANature& nature,
          numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
          const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::vi_gs(algorithms::SARobustOutcomeBellman(mdp, nature, policy),
                             discount, move(valuefunction), iterations, maxresidual,
                             progress);
}

/**
 * @ingroup PolicyIteration
 * Robust policy iteration with an s,a-rectangular nature.
 *
 * WARNING: This method is not guaranteed to converge to the optimal solution
 * or converge at all. The divergence of the method is shown in:
 * Condon, A. (1993). On algorithms for simple stochastic games.
 * Advances in Computational Complexity Theory, DIMACS Series in Discrete Mathematics
 * and Theoretical Computer Science, 13, 51–71.
 */
inline SARobustSolution
rsolve_pi(const MDPO& mdp, prec_t discount, const algorithms::SANature& nature,
          numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
          unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
          const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::pi(algorithms::SARobustOutcomeBellman(mdp, nature, policy),
                          discount, move(valuefunction), iterations, maxresidual,
                          progress);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s,a-rectangular nature.
 *
 * WARNING: There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 */
inline SARobustSolution rsolve_mpi(
    const MDPO& mdp, prec_t discount, const algorithms::SANature&& nature,
    const numvec& valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
    unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::mpi_jac(algorithms::SARobustOutcomeBellman(mdp, nature, policy),
                               discount, valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 * Robust partial policy iteration with an s,a-rectangular nature.
 *
 * This method is guaranteed to converge to the optimal value function
 * and policy.
 *
 * Uses  policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 */
inline SARobustSolution rsolve_ppi(
    const MDPO& mdp, prec_t discount, const algorithms::SANature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    return algorithms::rppi(algorithms::SARobustOutcomeBellman(mdp, nature, policy),
                            discount, move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::pi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 * Robust partial policy iteration with an s,a-rectangular nature.
 *
 * This method is guaranteed to converge to the optimal value function
 * and policy.
 *
 * Uses modified policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 */
inline SARobustSolution rsolve_mppi(
    const MDPO& mdp, prec_t discount, const algorithms::SANature& nature,
    numvec valuefunction = numvec(0), const indvec& policy = indvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    return algorithms::rppi(algorithms::SARobustOutcomeBellman(mdp, nature, policy),
                            discount, move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::mpi, progress);
}

// **************************************************************************
// Robust S-Rect MDPO methods
// **************************************************************************

/**
 * @ingroup ValueIteration
 * Robust value iteration with an s-rectangular nature.
 */
inline SRobustOutcomeSolution rsolve_s_vi(
    const MDPO& mdp, prec_t discount, const algorithms::SNatureOutcome& nature,
    numvec valuefunction = numvec(0), const numvecvec& policy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::vi_gs(algorithms::SRobustOutcomeBellman(mdp, nature, policy),
                             discount, move(valuefunction), iterations, maxresidual,
                             progress);
}

/**
 * @ingroup PolicyIteration
 * Robust policy iteration with an s-rectangular nature.
 *
 * WARNING: This method is not guaranteed to converge to the optimal solution
 * or converge at all. The divergence of the method is shown in:
 * Condon, A. (1993). On algorithms for simple stochastic games.
 * Advances in Computational Complexity Theory, DIMACS Series in Discrete Mathematics
 * and Theoretical Computer Science, 13, 51–71.
 */
inline SRobustOutcomeSolution rsolve_s_pi(
    const MDPO& mdp, prec_t discount, const algorithms::SNatureOutcome& nature,
    numvec valuefunction = numvec(0), const numvecvec& policy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::pi(algorithms::SRobustOutcomeBellman(mdp, nature, policy),
                          discount, move(valuefunction), iterations, maxresidual,
                          progress);
}

/**
 * @ingroup ModifiedPolicyIteration
 *
 * Robust modified policy iteration with an s-rectangular nature.
 *
 * WARNING: There is no proof of convergence for this method. This is not the
 * same algorithm as in: Kaufman, D. L., & Schaefer, A. J. (2013). Robust
 * modified policy iteration. INFORMS Journal on Computing, 25(3), 396–410. See
 * the discussion in the paper on methods like this one (e.g. Seid, White)
 */
inline SRobustOutcomeSolution rsolve_s_mpi(
    const MDPO& mdp, prec_t discount, const algorithms::SNatureOutcome&& nature,
    const numvec& valuefunction = numvec(0), const numvecvec& policy = numvecvec(0),
    unsigned long iterations_pi = MAXITER, prec_t maxresidual_pi = SOLPREC,
    unsigned long iterations_vi = MAXITER, prec_t maxresidual_vi = 0.9,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);
    return algorithms::mpi_jac(algorithms::SRobustOutcomeBellman(mdp, nature, policy),
                               discount, valuefunction, iterations_pi, maxresidual_pi,
                               iterations_vi, maxresidual_vi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 * Robust partial policy iteration with an s-rectangular nature.
 *
 * This method is guaranteed to converge to the optimal value function
 * and policy.
 *
 * Uses  policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 */
inline SRobustOutcomeSolution rsolve_s_ppi(
    const MDPO& mdp, prec_t discount, const algorithms::SNatureOutcome& nature,
    numvec valuefunction = numvec(0), const numvecvec& policy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {
    check_model(mdp);

    return algorithms::rppi(algorithms::SRobustOutcomeBellman(mdp, nature, policy),
                            discount, move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::pi, progress);
}

/**
 * @ingroup PartialPolicyIteration
 * Robust partial policy iteration with an s-rectangular nature.
 *
 * This method is guaranteed to converge to the optimal value function
 * and policy.
 *
 * Uses modified policy iteration to solve the inner MDP problem that corresponds
 * to the nature.
 */
inline SRobustOutcomeSolution rsolve_s_mppi(
    const MDPO& mdp, prec_t discount, const algorithms::SNatureOutcome& nature,
    numvec valuefunction = numvec(0), const numvecvec& policy = numvecvec(0),
    unsigned long iterations = MAXITER, prec_t maxresidual = SOLPREC,
    const algorithms::progress_t& progress = algorithms::internal::empty_progress) {

    check_model(mdp);

    return algorithms::rppi(algorithms::SRobustOutcomeBellman(mdp, nature, policy),
                            discount, move(valuefunction), iterations, maxresidual, 1.0,
                            discount * discount, algorithms::MDPSolver::mpi, progress);
}

// **************************************************************************
// Compute Occupancy Frequency
// **************************************************************************

} // namespace craam
