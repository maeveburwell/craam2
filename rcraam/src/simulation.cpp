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

#include "utils.hpp"
#include <array>

#include "craam/Samples.hpp"
#include "craam/simulation.hpp"
#include "craam/simulators/inventory.hpp"
#include "craam/simulators/population.hpp"

/**
 * A very simple test MDP.
 */
craam::MDP create_test_mdp() {
    craam::MDP rmdp(3);

    // nonrobust and deterministic
    // action 1 is optimal, with transition matrix [[0,1,0],[0,0,1],[0,0,1]] and rewards [0,0,1.1]
    // action 0 has a transition matrix [[1,0,0],[1,0,0], [0,1,0]] and rewards [0,1.0,1.0]
    add_transition(rmdp, 0, 1, 1, 1.0, 0.0);
    add_transition(rmdp, 1, 1, 2, 1.0, 0.0);
    add_transition(rmdp, 2, 1, 2, 1.0, 1.1);

    add_transition(rmdp, 0, 0, 0, 1.0, 0.0);
    add_transition(rmdp, 1, 0, 0, 1.0, 1.0);

    add_transition(rmdp, 2, 0, 1, 1.0, 1.0);

    return rmdp;
}

/**
 * Creates a simple test MDP
 */
//[[Rcpp::export]]
Rcpp::DataFrame mdp_example(Rcpp::String name) {
    return mdp_to_dataframe(create_test_mdp());
}

/**
 * Returns an optional value from a list and a default value if the option is not present.
 */
template <class Type>
inline Type getopt(const Rcpp::List& list, const std::string& name, Type&& defval) {
    return list.containsElementNamed(name.c_str()) ? Rcpp::as<Type>(list[name])
                                                   : std::forward(defval);
}

/**
 * Creates an inventory MDP description
 *
 * @param parameters Parameters describing the inventory management
 *              problem
 */
//[[Rcpp::export]]
Rcpp::DataFrame mdp_inventory(Rcpp::List params) {
    double purchase_cost = Rcpp::as<double>(params["variable_cost"]),
           fixed_cost = Rcpp::as<double>(params["fixed_cost"]),
           holding_cost = Rcpp::as<double>(params["holding_cost"]),
           backlog_cost = Rcpp::as<double>(params["backlog_cost"]),
           sale_price = Rcpp::as<double>(params["sale_price"]);
    long max_inventory = Rcpp::as<long>(params["max_inventory"]),
         max_backlog = Rcpp::as<long>(params["max_backlog"]),
         max_order = Rcpp::as<long>(params["max_order"]);
    craam::numvec demand_probabilities = Rcpp::as<craam::numvec>(params["demands"]);
    long rand_seed = Rcpp::as<long>(params["seed"]);

    std::array<double, 4> costs{purchase_cost, fixed_cost, holding_cost, backlog_cost};
    std::array<long, 3> limits{max_inventory, max_backlog, max_order};

    craam::msen::InventorySimulator simulator(demand_probabilities, costs, sale_price,
                                              limits);
    simulator.set_seed(rand_seed);

    craam::MDP fullmdp;

    simulator.build_mdp(
        [&fullmdp](long statefrom, long action, long stateto, double prob, double rew) {
            add_transition(fullmdp, statefrom, action, stateto, prob, rew);
        });
    return mdp_to_dataframe(fullmdp);
}

/**
 * Converts a numeric matrix to a vector of vectors. The rows translate
 * to the outer vector.
 */
craam::numvecvec matrix2nestedvec(const Rcpp::NumericMatrix& matrix) {
    craam::numvecvec result;
    result.reserve(matrix.nrow());
    for (int i = 0; i < matrix.nrow(); i++) {
        auto row = matrix.row(i);
        result.emplace_back(row.size());
        std::copy(row.cbegin(), row.cend(), result.back().begin());
    }
    return result;
}

/**
 * Creates a population model MDP
 */
//[[Rcpp::export]]
Rcpp::DataFrame mdp_population(int capacity, int initial,
                               Rcpp::NumericMatrix growth_rates_exp,
                               Rcpp::NumericMatrix growth_rates_std,
                               Rcpp::NumericMatrix rewards, double external_mean,
                               double external_std, Rcpp::String s_growth_model) {

    craam::msen::PopulationSim::Growth growth;

    if (s_growth_model == "logistic") {
        growth = craam::msen::PopulationSim::Growth::Logistic;
    } else if (s_growth_model == "exponential") {
        growth = craam::msen::PopulationSim::Growth::Exponential;
    } else {
        Rcpp::stop("Unknown growth model. Use 'exponential' or 'logistic'.");
        return Rcpp::DataFrame();
    }

    // use the logistic growth model by default
    auto sim = craam::msen::PopulationSim(
        capacity, initial, growth_rates_exp.nrow(), matrix2nestedvec(growth_rates_exp),
        matrix2nestedvec(growth_rates_std), matrix2nestedvec(rewards), external_mean,
        external_std, growth);

    craam::MDP mdp = craam::msen::build_mdp(sim, 10000);

    return mdp_to_dataframe(mdp);
}

/**
 * Converts samples from a simulation to a dataframe
 */
Rcpp::DataFrame samples_to_dtf(const craam::msen::DiscreteSamples& samples) {
    return Rcpp::DataFrame::create(
        Rcpp::_["idstatefrom"] = as_intvec(samples.get_states_from()),
        Rcpp::_["idaction"] = as_intvec(samples.get_actions()),
        Rcpp::_["idstateto"] = as_intvec(samples.get_states_to()),
        Rcpp::_["reward"] = samples.get_rewards(), Rcpp::_["step"] = samples.get_steps(),
        Rcpp::_["episode"] = samples.get_runs());
}

//'
//' Simulates an MDP
//'
//' @param mdp Definition of the MDP
//' @param initial_state The index of the initial state
//' @param policy Assumes a randomized policy as the input
//' @param horizon How many steps to execute for each episode
//' @param episodes Number of episodes to run
//' @param seed Random number generator seed (a number)
//'
//' @return A dataframe with the states, actions, and rewards received
//'
// [[Rcpp::export]]
Rcpp::DataFrame simulate_mdp(Rcpp::DataFrame mdp, int initial_state,
                             Rcpp::DataFrame policy, int horizon, int episodes,
                             Rcpp::Nullable<long> seed = R_NilValue) {

    craam::MDP m = mdp_from_dataframe(mdp);
    craam::numvecvec rpolicy_par = parse_sa_values(m, policy, 0.0, "probability");

    //Rcpp::Rcout << rpolicy_par[0][0] << std::endl;

    // initial transition probability
    craam::Transition init;
    init.add_sample(initial_state, 1.0, 0.0);

    // construct the simulator with the given random seed
    std::random_device::result_type cseed =
        seed.isSet() ? Rcpp::as<long>(seed.get()) : std::random_device{}();

    auto sim_model =
        craam::msen::ModelSimulator(std::make_shared<const craam::MDP>(m), init, cseed);

    craam::msen::RandomizedPolicy<craam::msen::ModelSimulator> rpolicy(
        sim_model, rpolicy_par, cseed);

    craam::msen::DiscreteSamples samples;

    craam::msen::simulate(sim_model, samples, rpolicy, horizon, episodes);

    return samples_to_dtf(samples);
}
