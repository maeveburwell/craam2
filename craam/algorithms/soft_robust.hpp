// This file is part of CRAAM, a C++ library for solving plain
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

// The file includes methods to solve MDPO with a soft-robust objective

#pragma once

#include "craam/definitions.hpp"

#ifdef GUROBI_USE

#include "craam/MDPO.hpp"
#include "craam/Solution.hpp"

#include <chrono>
#include <memory>

#include <gurobi/gurobi_c++.h>

namespace craam::statalgs {

/**
 * Solves a MDPO (uncertain MDP) with a AVaR soft-robust objective, assuming
 * static uncertainty.
 *
 *
 *
 * The problem is formulated as a *non-convex* quadratic program and solved
 * using Gurobi. The objective to solve is:
 * max_pi lambda * CVaR_{P ~ f}^alpha [return(pi,P)] +
 *        (1-lambda) * E_{P ~ f}^alpha [return(pi,P)]
 * where pi is a randomized policy. The formulation allows for uncertain rewards
 * jointly with uncertain transition probabilities.
 *
 * The outcomes in the MDPO are represented by the value omega below (Omega
 * is the set of all possible outcomes.) The states are s, and actions are a.
 *
 * The actual quadratic program formulation is as follows:
 * max_{pi, d} max_{z,y} z + sum_{omega} (
 *              (1-beta) * sum_{s,a} d(s,omega) pi(s,a) r^omega(s,a) -
 *                 beta  * y(omega) )
 * subject to:
 *      y(omega) >= z - sum_{s,a} d(s,omega) pi(s,a) r^omega(s,a) for each omega
 *      d(s,omega) = gamma * sum_{s',a'} d(s',omega) * pi(s',a') * P^omega(s',a',s) +
 *                      f(omega) p_0(s), for each omega and s
 *      sum_a pi(s,a) = 1  for each s
 *      pi(s,a) >= 0 for each s and a
 *      d(s,omega) >= 0 for each s and omega
 * Here, p0 is the initial distribution and f is the nominal distribution over the models
 *
 * @param mdpo Uncertain MDP. The outcomes are assumed to represent the uncertainty over MDPs.
 *              The number of outcomes must be uniform for all states and actions
 *              (except for terminal states which have no actions).
 * @param alpha Risk level of avar (0 = worst-case). The minimum value is 1e-5, the maximum
 *              value is 1.
 * @param beta Weight on AVaR and the complement (1-beta) is the weight
 *              on the expectation term. The value must be between 0 and 1.
 * @param gamma Discount factor. Clamped to be in [0,1]
 * @param init_distribution Initial distribution over states
 * @param model_dist Distribution over the models. The default is empty, which translates
 *                   to a uniform distribution.
 * @return Solution that includes the policy and the value function
 */
inline RandStaticSolution srsolve_avar_quad(const GRBEnv& env, const MDPO& mdpo,
                                            prec_t alpha, prec_t beta, prec_t gamma,
                                            const ProbDst& init_dist,
                                            const ProbDst& model_dist = ProbDst(0)) {
    const prec_t alpha_min = 1e-5;

    // general constants values
    const double inf = std::numeric_limits<prec_t>::infinity();
    const size_t nstates = mdpo.size();

    if (nstates == 0)
        return {.objective = 0, .time = 0, .status = 0, .message = "Empty MDPO"};

    if (init_dist.size() != nstates)
        throw std::invalid_argument("The initial distribution init_dist must have the "
                                    "same length as the number of states.");
    // check if the number of outcomes is the same for all states and actions
    // TODO: move this functionality to the MDPO definition?
    // count the number of outcomes
    const size_t noutcomes = [&] {
        // (find a non-terminal state first)
        auto ps = std::find_if_not(mdpo.begin(), mdpo.end(),
                                   [&](const StateO& s) { return s.is_terminal(); });
        // not found: -1
        return ps == mdpo.end() ? -1ul : ps->get_action(0).outcome_count();
    }();

    // all states are terminal, just return an empty solution
    if (noutcomes < 0)
        return {
            .objective = 0, .time = 0, .status = 0, .message = "All states are terminal"};

    // find a state with outcomes that do not match the expected number
    for (size_t is = 0; is < mdpo.size(); ++is)
        for (size_t ia = 0; ia < mdpo[is].size(); ++ia)
            if (mdpo[is][ia].size() != noutcomes)
                throw ModelError(
                    "Number of outcomes is not uniform across all states and actions", is,
                    ia);

    if (!model_dist.empty() && model_dist.size())
        throw std::invalid_argument("Model distribution must either be empty or have the "
                                    "same length as the number of outcomes.");
    // assume a uniform distribution if not provided
    const prec_t model_dist_const = 1.0 / prec_t(noutcomes);

    // time the computation
    auto start = chrono::steady_clock::now();

    // --- clamp input values to between 0 and 1
    alpha = std::clamp(alpha, alpha_min, alpha);
    beta = std::clamp(beta, 0.0, 1.0);
    gamma = std::clamp(gamma, 0.0, 1.0);

    // --- proceed with creating the model -----
    GRBModel model(env);

    // build a vector of vectors with each element referencing the index in the policy
    // cannot use auto [..,..] because the reference does not work with the lambda later
    vector<indvec> array_index_sa;
    size_t nstateactions;
    std::tie(array_index_sa, nstateactions) = [&]() {
        vector<indvec> index_sa(nstates);
        size_t count_sa = 0; // current count of state-action pairs
        for (size_t i = 0; i < nstates; ++i) {
            index_sa[i].resize(mdpo[i].size());
            std::iota(index_sa[i].begin(), index_sa[i].end(), count_sa);
            count_sa += mdpo[i].size();
        }
        return std::make_pair(move(index_sa), count_sa);
    }();
    // used to index pi
    const auto index_sa = [&](size_t s, size_t a) {
        assert(s < nstates && a < mdpo[s].size());
        return array_index_sa[s][a];
    };
    // used to index d, w is omega (first loop over s and then omega/w)
    const auto index_sw = [&](size_t s, size_t w) {
        assert(s < nstates && w < noutcomes);
        return s * noutcomes + w;
    };
    const size_t nstateoutcomes = nstates * noutcomes;

    auto pi = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(nstateactions, 0).data(), nullptr, nullptr,
        std::vector<char>(nstateactions, GRB_CONTINUOUS).data(), nullptr, nstateactions));

    auto d = std::unique_ptr<GRBVar[]>(
        model.addVars(numvec(nstateoutcomes, 0).data(), nullptr, nullptr,
                      std::vector<char>(nstateoutcomes, GRB_CONTINUOUS).data(), nullptr,
                      nstateoutcomes));

    auto y = std::unique_ptr<GRBVar[]>(model.addVars(
        numvec(noutcomes, 0).data(), nullptr, nullptr,
        std::vector<char>(noutcomes, GRB_CONTINUOUS).data(), nullptr, noutcomes));

    auto z = model.addVar(-inf, +inf, 0, GRB_CONTINUOUS, "");

    // objective: z + sum_{omega} (
    //              (1-beta) * sum_{s,a} d(s,omega) pi(s,a) r^omega(s,a) -
    //                beta * y(omega) )
    GRBQuadExpr objective = z;
    // TODO: could be faster using GRBQuadExpr::addTerms
    for (size_t iw = 0; iw < noutcomes; ++iw) {
        for (size_t is = 0; is < nstates; ++is) {
            const StateO& s = mdpo[is];
            for (size_t ia = 0; ia < s.size(); ia++) {
                const auto reward = s[ia][iw].mean_reward();
                objective +=
                    (1 - beta) * reward * d[index_sw(is, iw)] * pi[index_sa(is, ia)];
            }
        }
        objective -= beta / (1.0 - alpha) * y[iw];
    }
    model.setObjective(objective, GRB_MAXIMIZE);
    /*
    // constraint:
    // y(omega) - z + sum_{s,a} d(s,omega) pi(s,a) r^omega(s,a) >= 0 for each omega
    for (size_t iw = 0; iw < noutcomes; ++iw) {
        GRBQuadExpr constraint_y = -z + y[iw];
        for (size_t is = 0; is < nstates; ++is) {
            const StateO& s = mdpo[is];
            for (size_t ia = 0; ia < s.size(); ia++) {
                const auto reward = s[ia][iw].mean_reward();
                constraint_y += reward * d[index_sw(is, iw)] * pi[index_sa(is, ia)];
            }
        }
        model.addQConstr(constraint_y >= 0);
    }

    // constraint:
    //d(s,omega) - gamma * sum_{s',a'} d(s',omega) * pi(s',a') * P^omega(s',a',s) =
    //                      f(omega) p_0(s), for each omega and s
    for (size_t iw = 0; iw < noutcomes; ++iw) {   // omega
        for (size_t is = 0; is < nstates; ++is) { // state s
            // d(s,omega)
            GRBQuadExpr constraint_d = d[index_sw(is, iw)];

            for (size_t isp = 0; isp < nstates; ++isp) { //state s'
                const StateO& sp = mdpo[isp];
                for (size_t iap = 0; iap < sp.size(); ++iap) { // action a'
                    const prec_t probability = sp[iap][iw].probability_to(is);
                    // skip of the probability is 0
                    if (probability > 0)
                        //- gamma * sum_{s',a'} d(s',omega) * pi(s',a') * P^omega(s',a',s)
                        constraint_d -= gamma * d[index_sw(isp, iw)] *
                                        pi[index_sa(isp, iap)] * probability;
                }
            }
            model.addQConstr(
                constraint_d ==
                init_dist[is] * (model_dist.empty() ? model_dist_const : model_dist[iw]));
        }
    }*/

    // constraint:
    // sum_a pi(s,a) = 1  for each s
    for (size_t is = 0; is < nstates; ++is) {
        GRBLinExpr constraint_pi;

        const StateO& s = mdpo[is];
        for (size_t ia = 0; ia < s.size(); ia++) {
            constraint_pi += pi[index_sa(is, ia)];
        }
        model.addConstr(constraint_pi == 1.0);
    }

#ifndef NDEBUG
    // Use to determine whether the solution is unbounded or infeasible
    // enabled when debugging but disabled
    model.set(GRB_IntParam_DualReductions, 0);
#endif

    // solve the optimization problem
    model.optimize();

    int status = model.get(GRB_IntAttr_Status);

    /*
     * LOADED       1 	Model is loaded, but no solution information is available.
     * OPTIMAL      2 	Model was solved to optimality (subject to tolerances),
     *                  and an optimal solution is available.
     * INFEASIBLE 	3 	Model was proven to be infeasible.
     * INF_OR_UNBD 	4 	Model was proven to be either infeasible or unbounded.
     *                  To obtain a more definitive conclusion, set
     *                  the DualReductions parameter to 0 and reoptimize.
     * UNBOUNDED 	5 	Model was proven to be unbounded. Important note:
     *                  an unbounded status indicates the presence of an unbounded ray
     *                  that allows the objective to improve without limit.
     *                  It says nothing about whether the model has a feasible solution.
     *                  If you require information on feasibility, you should set the
     *                  objective to zero and reoptimize.
     * CUTOFF       6 	Optimal objective for model was proven to be worse than
     *                  the value specified in the Cutoff parameter.
     *                  No solution information is available.
     */
    if (status != GRB_OPTIMAL) {
        return {.message = "Solution infeasible or unbounded."};
        //throw runtime_error("Failed to solve the optimization problem.");
    }

    numvecvec policy{nstates};
    for (size_t is = 0; is < nstates; ++is) {
        const StateO& s = mdpo[is];
        policy[is].resize(s.size());
        for (size_t ia = 0; ia < s.size(); ++ia) {
            policy[is][ia] = pi[index_sa(is, ia)].get(GRB_DoubleAttr_X);
        }
    }

    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish - start;
    return {.policy = move(policy),
            .objective = objective.getValue(),
            .time = duration.count(),
            .status = 0};
}

} // namespace craam::statalgs

#endif
