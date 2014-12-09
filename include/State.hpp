#ifndef STATE_H
#define STATE_H

#include <utility>
#include <tuple>
#include <vector>
#include <stdexcept>

#include "Action.hpp"

using namespace std;

class State {
public:
    vector<Action> actions;

    State(){};
    State(vector<Action> actions){
        this->actions = actions;
    }

    tuple<long,long,prec_t> max_max(vector<prec_t> const& valuefunction, prec_t discount) const;
    tuple<long,long,prec_t> max_min(vector<prec_t> const& valuefunction, prec_t discount) const;

    tuple<long,vector<prec_t>,prec_t> max_max_l1(vector<prec_t> const& valuefunction, prec_t discount) const;
    tuple<long,vector<prec_t>,prec_t> max_min_l1(vector<prec_t> const& valuefunction, prec_t discount) const;

    pair<long,prec_t> max_average(vector<prec_t> const& valuefunction, prec_t discount) const;

    // functions used in modified policy iteration
    prec_t fixed_average(vector<prec_t> const& valuefunction, prec_t discount, long actionid, vector<prec_t> const& distribution) const;
    prec_t fixed_average(vector<prec_t> const& valuefunction, prec_t discount, long actionid) const;
    prec_t fixed_fixed(vector<prec_t> const& valuefunction, prec_t discount, long actionid, long outcomeid) const;

    void add_action(long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);

    void set_thresholds(prec_t threshold);
};

#endif // STATE_H