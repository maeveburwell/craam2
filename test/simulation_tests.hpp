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

#pragma once

#include "craam/GMDP.hpp"
#include "craam/algorithms/values.hpp"
#include "craam/modeltools.hpp"
#include "craam/simulation.hpp"
#include "craam/simulators/inventory.hpp"
#include "craam/simulators/population.hpp"

#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>

#include <boost/functional/hash.hpp>

using namespace std;
using namespace craam;
using namespace craam::msen;
using namespace craam::algorithms;
using namespace util::lang;

struct TestState {
    int index;

    TestState(int i) : index(i){};
};

class TestSim {

public:
    typedef TestState State;
    typedef int Action;

    TestState init_state() const { return TestState(1); }

    pair<double, TestState> transition(TestState, int) const {
        return pair<double, TestState>(1.0, TestState(1));
    };

    bool end_condition(TestState) const { return false; };

    long action_count(TestState) const { return 1; }

    int action(TestState, long) const { return 1; }
};

int test_policy(TestState) { return 0; }

BOOST_AUTO_TEST_CASE(basic_simulation) {
    TestSim sim;

    auto samples = simulate<TestSim>(sim, test_policy, 10, 5);
    BOOST_CHECK_EQUAL(samples.size(), 50);
}

/**
A simple simulator class. The state represents a position in a chain
and actions move it up and down. The reward is equal to the position.

Representation
~~~~~~~~~~~~~~
- State: position (int)
- Action: change (int)
*/
class Counter {
private:
    default_random_engine gen;
    bernoulli_distribution d;
    const vector<int> actions_list;
    const int initstate;

public:
    using State = int;
    using Action = int;

    /**
  Define the success of each action
  \param success The probability that the action is actually applied
  */
    Counter(double success, int initstate,
            random_device::result_type seed = random_device{}())
        : gen(seed), d(success), actions_list({1, -1}), initstate(initstate) {}

    int init_state() const { return initstate; }

    pair<double, int> transition(int pos, int action) {
        int nextpos = d(gen) ? pos + action : pos;
        return make_pair(double(pos), nextpos);
    }

    bool end_condition(const int) { return false; }

    int action(State, long index) const { return actions_list[size_t(index)]; }

    virtual vector<int> get_valid_actions(State state) const { return actions_list; }

    size_t action_count(State) const { return actions_list.size(); }

    virtual ~Counter() {}
};

/** A counter that terminates at either end as defined by the end state */
class CounterTerminal : public Counter {
public:
    int endstate;

    CounterTerminal(double success, int initstate, int endstate,
                    random_device::result_type seed = random_device{}())
        : Counter(success, initstate, seed), endstate(endstate){};

    bool end_condition(const int state) { return (abs(state) >= endstate); }

    virtual ~CounterTerminal() {}
};

/** A counter that has bound states */
class CounterBound : public Counter {
private:
    int min_state;
    int max_state;
    vector<int> min_state_actions;
    vector<int> max_state_actions;

public:
    CounterBound(double success, int init_state, int min_state, int max_state,
                 random_device::result_type seed = random_device{}())
        : Counter(success, init_state, seed), min_state(min_state), max_state(max_state),
          min_state_actions({1}), max_state_actions({-1}){};

    virtual vector<int> get_valid_actions(State state) const {
        if (state == min_state)
            return min_state_actions;
        else if (state == max_state)
            return max_state_actions;
        else
            return Counter::get_valid_actions(state);
    }
    virtual ~CounterBound() {}
};

// Hash function for the Counter / CounterTerminal EState above
namespace std {
template <> struct hash<pair<int, int>> {
    size_t operator()(pair<int, int> const& s) const {
        boost::hash<pair<int, int>> h;
        return h(s);
    };
};
} // namespace std

BOOST_AUTO_TEST_CASE(simulation_multiple_counter_si) {
    Counter sim(0.9, 0, 1);

    RandomPolicy<Counter> random_pol(sim, 1);
    auto samples = simulate(sim, random_pol, 20, 20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), -3.51759102217019, 0.0001);

    samples = simulate(sim, random_pol, 1, 20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 0, 0.0001);

    Counter sim2(0.9, 3, 1);
    samples = simulate(sim2, random_pol, 1, 20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 3, 0.0001);
}

BOOST_AUTO_TEST_CASE(simulation_multiple_counter_si_return) {
    Counter sim(0.9, 0, 1);

    RandomPolicy<Counter> random_pol(sim, 1); // sets the random seed
    auto samples_returns = simulate_return(sim, 0.9, random_pol, 20, 20);
    auto v = samples_returns.second;
    auto meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, -3.51759102217019, 0.0001);

    samples_returns = simulate_return(sim, 0.9, random_pol, 1, 20);
    v = samples_returns.second;
    meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, 0, 0.0001);

    Counter sim2(0.9, 3, 1);
    samples_returns = simulate_return(sim2, 0.9, random_pol, 1, 20);
    v = samples_returns.second;
    meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, 3, 0.0001);

    // test termination probability equals to discount
    samples_returns = simulate_return(
        sim, 1.0, DeterministicPolicy<Counter>(sim, indvec{0, 1, 1}), 1000, 100, 0.1, 1);
    v = samples_returns.second;
    meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, 4.73684, 10.0);

    // test stochastic policies
    CounterBound stochastic_sim(0.9, 0, 0, 2, 1);
    prob_matrix_t prob_matrix;
    prob_list_t prob_list1;
    prob_list_t prob_list2;
    prob_list_t prob_list3;
    prob_list1.push_back(1);
    prob_list1.push_back(0);
    prob_list2.push_back(0.6);
    prob_list2.push_back(0.4);
    prob_list3.push_back(0);
    prob_list3.push_back(1);
    prob_matrix.push_back(prob_list1);
    prob_matrix.push_back(prob_list2);
    prob_matrix.push_back(prob_list3);
    samples_returns =
        simulate_return(stochastic_sim, 1.0,
                        StochasticPolicy<Counter>(sim, prob_matrix, 1), 1000, 4, 0.1, 1);
    v = samples_returns.second;
    meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_EQUAL(meanreturn, 3);
}

BOOST_AUTO_TEST_CASE(cumulative_rewards) {
    // check that the reward is constructed correctly from samples
    DiscreteSamples samples;

    samples.add_sample(0, 0, 1, 1.0, 3.0, 0, 0);
    samples.add_sample(0, 0, 1, 2.0, 2.0, 0, 0);
    samples.add_sample(0, 0, 1, 3.0, 1.0, 0, 0);

    samples.add_sample(0, 0, 2, 7.0, 1.0, 0, 1);
    samples.add_sample(0, 0, 3, 2.0, 1.0, 0, 1);
    samples.add_sample(0, 0, 0, 0.0, 1.0, 0, 1);

    samples.add_sample(DiscreteSample(0, 0, 1, 1, 1, 0, 2));
    samples.add_sample(DiscreteSample(0, 0, 1, 11, 1, 0, 2));

    BOOST_CHECK_EQUAL(samples.get_cumulative_rewards()[2], 6);
    BOOST_CHECK_EQUAL(samples.get_cumulative_rewards()[5], 9);
    BOOST_CHECK_EQUAL(samples.get_cumulative_rewards()[7], 12);
}

BOOST_AUTO_TEST_CASE(sampled_mdp_reward) {
    // check that the reward is constructed correctly from samples
    DiscreteSamples samples;

    // relevant samples (transition to 1)
    samples.add_sample(0, 0, 1, 1.0, 3.0, 0, 0);
    samples.add_sample(0, 0, 1, 2.0, 2.0, 0, 0);
    samples.add_sample(0, 0, 1, 3.0, 1.0, 0, 0);
    // irrelevant samples (do not transition to 1)
    samples.add_sample(0, 0, 2, 0.0, 1.0, 0, 0);
    samples.add_sample(0, 0, 3, 0.0, 1.0, 0, 0);
    samples.add_sample(0, 0, 0, 0.0, 1.0, 0, 0);

    SampledMDP smdp;

    smdp.add_samples(samples);
    auto reward = (*smdp.get_mdp())[0][0].get_rewards()[1];
    // cout << (*smdp.get_mdp())[0][0].get_rewards()[1] << endl;

    BOOST_CHECK_CLOSE(reward, 1.666666, 1e-4);

    // check that the reward is constructed correctly from samples
    DiscreteSamples samples2;

    // relevant samples (transition to 1)
    samples2.add_sample(0, 0, 1, 2.0, 9.0, 0, 0);
    samples2.add_sample(0, 0, 1, 4.0, 6.0, 0, 0);
    samples2.add_sample(0, 0, 1, 6.0, 3.0, 0, 0);
    // irrelevant samples (do not transition to 1)
    samples2.add_sample(0, 0, 2, 0.0, 1.0, 0, 0);
    samples2.add_sample(0, 0, 3, 0.0, 1.0, 0, 0);
    samples2.add_sample(0, 0, 0, 0.0, 1.0, 0, 0);

    smdp.add_samples(samples2);
    // cout << (*smdp.get_mdp())[0][0].get_rewards()[1] << endl;
    reward = (*smdp.get_mdp())[0][0].get_rewards()[1];
    BOOST_CHECK_CLOSE(reward, 2.916666666666, 1e-4);
}

BOOST_AUTO_TEST_CASE(construct_mdp_from_samples_si_pol) {

    CounterTerminal sim(0.9, 0, 10, 1);
    RandomPolicy<CounterTerminal> random_pol(sim, 1);

    auto samples = make_samples<CounterTerminal>();
    simulate(sim, samples, random_pol, 50, 50);
    simulate(
        sim, samples, [](int) { return 1; }, 10, 20);
    simulate(
        sim, samples, [](int) { return -1; }, 10, 20);

    SampleDiscretizerSD<typename CounterTerminal::State, typename CounterTerminal::Action>
        sd;
    sd.add_samples(samples);

    BOOST_CHECK_EQUAL(samples.get_initial().size(),
                      sd.get_discrete()->get_initial().size());
    BOOST_CHECK_EQUAL(samples.size(), sd.get_discrete()->size());

    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());

    shared_ptr<const MDP> mdp = smdp.get_mdp();

    // check that the number of actions is correct (2)
    for (size_t i = 0; i < mdp->state_count(); ++i) {
        if (!(*mdp)[i].empty()) BOOST_CHECK_EQUAL((*mdp)[i].size(), 2);
    }

    auto sol = solve_mpi(*mdp, 0.9);

    BOOST_CHECK_CLOSE(sol.total_return(smdp.get_initial()), 51.313973, 1e-3);
}

template <class Model> Model create_test_mdp_sim() {
    Model rmdp(3);

    // nonrobust
    // action 1 is optimal, with transition matrix [[0,1,0],[0,0,1],[0,0,1]] and
    // rewards [0,0,1.1]
    add_transition(rmdp, 0, 1, 1, 1.0, 0.0);
    add_transition(rmdp, 1, 1, 2, 1.0, 0.0);
    add_transition(rmdp, 2, 1, 2, 1.0, 1.1);

    add_transition(rmdp, 0, 0, 0, 1.0, 0.0);
    add_transition(rmdp, 1, 0, 0, 1.0, 1.0);
    add_transition(rmdp, 2, 0, 1, 1.0, 1.0);

    add_transition(rmdp, 1, 2, 1, 0.5, 0.5);
    add_transition(rmdp, 1, 2, 3, 0.5, 0.0);

    return rmdp;
}

#if __cplusplus >= 201703L

BOOST_AUTO_TEST_CASE(simulate_mdp) {

    shared_ptr<MDP> m = make_shared<MDP>();
    *m = create_test_mdp_sim<MDP>();

    Transition initial({0}, {1.0});

    ModelSimulator ms(m, initial, 13);
    ModelRandomPolicy rp(ms, 10);

    auto samples = simulate(ms, rp, 1000, 5, -1, 0.0, 10);

    BOOST_CHECK_EQUAL(samples.size(), 49);
    // cout << "Number of samples " << samples.size() << endl;

    SampledMDP smdp;
    smdp.add_samples(samples);

    auto newmdp = smdp.get_mdp();

    auto solution1 = solve_mpi(*m, 0.9);
    auto solution2 = solve_mpi(*newmdp, 0.9);

    BOOST_CHECK_CLOSE(solution1.total_return(initial), 8.90971, 1.0);
    // cout << "Return in original MDP " << solution1.total_return(initial) <<
    // endl;

    BOOST_CHECK_CLOSE(solution2.total_return(initial), 8.90971, 1.0);
    // cout << "Return in sampled MDP " << solution2.total_return(initial) <<
    // endl;

    indvec policy = solution2.policy;

    indvec policytarget{1, 1, 1, -1};
    BOOST_CHECK_EQUAL_COLLECTIONS(policy.cbegin(), policy.cend(), policytarget.begin(),
                                  policytarget.end());
    auto solution3 = solve_mpi(*m, 0.9, numvec(0), policy);

    BOOST_CHECK_CLOSE(solution3.total_return(initial), 8.90916, 1e-2);
    // cout << "Return of sampled policy in the original MDP " <<
    // solution3.total_return(initial) << endl;

    ModelDeterministicPolicy dp(ms, policy);
    auto samples_policy = simulate(ms, dp, 1000, 5);

    BOOST_CHECK_CLOSE(samples_policy.mean_return(0.9), 8.91, 1e-3);
    // cout << "Return of sampled " << samples_policy.mean_return(0.9) << endl;

    ModelRandomizedPolicy rizedp(ms, {{0.5, 0.5}, {0.5, 0.4, 0.1}, {0.5, 0.5}}, 0);

    auto randomized_samples = simulate(ms, rizedp, 1000, 5, -1, 0.0, 10);

    BOOST_CHECK_CLOSE(randomized_samples.mean_return(0.9), 4.01147, 1e-3);
    // cout << "Return of randomized samples " <<
    // randomized_samples.mean_return(0.9) << endl;
}
#endif // _cplusplus >= 201703L

BOOST_AUTO_TEST_CASE(inventory_simulator) {
    // make sure that solving an MDP constructed from simulation and samples
    // returns the same solution as the MDP that is constructed directly

    long horizon = 100, num_runs = 500, initial = 0, max_inventory = 15;
    long rand_seed = 7;
    prec_t purchase_cost = 2, sale_price = 3;
    double discount = 0.9;

    numvec demand_probabilities{0.1, 0.2, 0.3, 0.3, 0.1};
    std::array<prec_t, 4> costs{purchase_cost, 0.0, 0.2, 0.1};
    std::array<long, 3> limits{max_inventory, 0l, 5l};

    InventorySimulator simulator(demand_probabilities, costs, sale_price, limits);
    simulator.set_seed(rand_seed);

#if __cplusplus >= 201703L
    RandomPolicy rp(simulator, rand_seed);
#else
    RandomPolicy<InventorySimulator> rp(simulator, rand_seed);
#endif
    // solve a sampled-version of the MDP
    auto samples = simulate(simulator, rp, horizon, num_runs, -1, 0.0, rand_seed);
    BOOST_CHECK_EQUAL(samples.size(), horizon * num_runs);

    SampledMDP smdp;
    smdp.add_samples(samples);
    // get a copy
    MDP newmdp = *smdp.get_mdp();
    newmdp.pack_actions(); // remove actions that have no samples
    auto solution = solve_mpi(newmdp, discount);
    Transition init({initial}, {1.0});

    // Need to know what the exact return should be to make the below test
    // meaningful.
    ///BOOST_CHECK_CLOSE(solution.total_return(init), 29.5768, 1e-2);

    // solve the generated version of the problem
    MDP fullmdp;
    simulator.build_mdp(
        [&fullmdp](long statefrom, long action, long stateto, prec_t prob, prec_t rew) {
            add_transition(fullmdp, statefrom, action, stateto, prob, rew);
        });

    auto solution2 = solve_mpi(fullmdp, discount);

    BOOST_CHECK_CLOSE(solution.total_return(init), solution2.total_return(init), 3.0);
}

BOOST_AUTO_TEST_CASE(inventory_build_mdp) {
    double purchase_cost = 2.49, fixed_cost = 5.99, holding_cost = 0.03,
           backlog_cost = 0.15, sale_price = 3.99;
    long max_inventory = 100, max_backlog = 10, max_order = 50;
    craam::numvec demand_probabilities =
        numvec{0.01119473, 0.01636988, 0.02299882, 0.03104516, 0.04026340, 0.05017129,
               0.06006593, 0.06909227, 0.07635877, 0.08108053, 0.08271847, 0.08108053,
               0.07635877, 0.06909227, 0.06006593, 0.05017129, 0.04026340, 0.03104516,
               0.02299882, 0.01636988, 0.01119473};
    long rand_seed = 1;

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

    BOOST_CHECK_EQUAL(fullmdp.size(), 111);
}

BOOST_AUTO_TEST_CASE(population_simulator) {
    long horizon = 10, num_runs = 5, initial_population = 30, carrying_capacity = 1000;
    int rand_seed = 7;
    uint threshold_control = 100;
    prec_t prob_control = 0.8;

    numvecvec mean_rate = {numvec(1 + carrying_capacity, 1.03),
                           numvec(1 + carrying_capacity, 0.95)};
    numvecvec std_rate = {numvec(1 + carrying_capacity, 0.5),
                          numvec(1 + carrying_capacity, 0.5)};
    numvecvec rewards = {numvec(1 + carrying_capacity, -1.0),
                         numvec(1 + carrying_capacity, 0.2)};

    PopulationSim simulator(carrying_capacity, initial_population, 2, mean_rate, std_rate,
                            rewards, 0, 0, PopulationSim::Growth::Exponential, rand_seed);
    PopulationPol rp(simulator, threshold_control, prob_control, rand_seed);

    auto samples = simulate(simulator, rp, horizon, num_runs, -1, 0.0, rand_seed);
    BOOST_CHECK_EQUAL(samples.size(), horizon * num_runs);

    SampledMDP smdp;
    smdp.add_samples(samples);
    MDP newmdp = *smdp.get_mdp();
    newmdp.pack_actions();
    auto solution = solve_mpi(newmdp, 0.9);
    Transition init({initial_population}, {1.0});

    // Need to know what the exact return should be to make the below test
    // meaningful.
    // TODO: enable a check here
    //BOOST_CHECK_CLOSE(solution.total_return(init), -0.4245, 1e-2);
}
