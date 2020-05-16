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

#include "Action.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace craam {

using namespace std;

// **************************************************************************************
//  SA State (SA rectangular, also used for a regular MDP)
// **************************************************************************************

/**
State for sa-rectangular uncertainty (or no uncertainty) in an MDP

A state with no actions is considered to be terminal and its value is 0.

Actions within a state are sequentially labeled. That is, adding an action with
an id = 3 will also create actions 0,1,2.

Any actions that have no transitions will be considered invalid and the MDP
computation will fail. Use a transition to a terminal state to model the end of
the execution.

@tparam Type of action used in the state. This type determines the
    type of uncertainty set.
*/
template <class AType> class SAState {
protected:
    /// list of actions
    vector<AType> actions;

public:
    SAState() : actions(0){};

    /** Initializes state with actions and sets them all to valid */
    SAState(vector<AType> actions) : actions(move(actions)){};

    /** Number of actions */
    size_t size() const { return actions.size(); };

    /**
  Creates an action given by actionid if it does not exists.
  Otherwise returns the existing one.

  All newly created actions with id < actionid are created as invalid
  (action.get_valid() = false). Action actionid is created as valid.
  */
    AType& create_action(long actionid) {
        assert(actionid >= 0);

        // assumes that the default constructor makes the actions invalid
        if (actionid >= (long)actions.size()) { actions.resize(actionid + 1); }

        // set only the action that is being added as valid
        return actions[actionid];
    }

    /** Creates an action at the last position of the state */
    AType& create_action() { return create_action(actions.size()); };

    /** Returns an existing action */
    const AType& operator[](long actionid) const { return get_action(actionid); }

    /** Returns an existing action */
    AType& operator[](long actionid) { return get_action(actionid); }

    /** Returns the set of all actions */
    const vector<AType>& get_actions() const { return actions; };

    /** Check whether it is empty */
    bool empty() const { return actions.empty(); }

    /** True if the state is considered terminal (no actions). */
    bool is_terminal() const { return actions.empty(); };

    /** Normalizes transition probabilities to sum to one. */
    void normalize() {
        for (AType& a : actions)
            a.normalize();
    }

    /** Checks whether the prescribed action correct */
    bool is_action_correct(long aid) const {
        if ((aid < 0) || ((size_t)aid >= actions.size()))
            return false;
        else
            return true;
    }

    /** Returns the mean transition probabilities following the action and
      outcome. This class assumes a deterministic policy of the decision maker and
      a randomized policy of nature.

      @param action Deterministic action of the decision maker
      @param nataction Randomized action of nature */
    Transition mean_transition(long action, numvec nataction) const {
        if (is_terminal())
            return Transition();
        else
            return get_action(action).mean_transition(nataction);
    }

    /** Returns the mean transition probabilities following the action and
      outcome.

      @param action Deterministic action of decision maker */
    Transition mean_transition(long action) const {
        if (is_terminal())
            return Transition();
        else
            return get_action(action).mean_transition();
    }

    /** Returns json representation of the state
        @param stateid Includes also state id*/
    string to_json(long stateid = -1) const {
        stringstream result;
        result << "{";
        result << "\"stateid\" : ";
        result << std::to_string(stateid);
        result << ",\"actions\" : [";
        for (auto ai : indices(actions)) {
            const auto& a = actions[ai];
            result << a.to_json(ai);
            result << ",";
        }
        //if (!actions.empty()) result.pop_back(); // remove last comma
        result << ("]}");
        return result.str();
    }

    /**
      * Removes invalid actions, and reindexes the remaining ones accordingly.
      *
      * This function is not thread-safe and could leave the object in a very bad
      * internal state if interrupted
      *
      * @returns List of original action ids which were selected from the list
      */
    indvec pack_actions() {
        indvec original;
        vector<AType> newactions;
        for (size_t actionid = 0; actionid < actions.size(); actionid++) {
            AType& action = actions[actionid];
            if (action.has_transitions()) {
                newactions.push_back(move(action));
                original.push_back(actionid);
            }
        }
        actions = move(newactions);
        return original;
    }

    // ******************************************
    // Collection functions
    // *******************************************

    auto begin() const { return actions.cbegin(); }
    auto end() const { return actions.cend(); }

    /** Returns an existing action */
    const AType& get_action(long actionid) const {
        assert(actionid >= 0 && size_t(actionid) < size());
        return actions[actionid];
    };

protected:
    /** Returns an existing action */
    AType& get_action(long actionid) {
        assert(actionid >= 0 && size_t(actionid) < size());
        return actions[actionid];
    };
};

// **********************************************************************
// *********************    SPECIFIC STATE DEFINITIONS    ***************
// **********************************************************************

/// helper functions
namespace internal {
using namespace craam;

/// checks state and policy with a policy of nature
template <class SType>
bool is_action_correct(const SType& state, long stateid,
                       const std::pair<indvec, vector<numvec>>& policies) {
    return state.is_action_correct(policies.first[stateid], policies.second[stateid]);
}

/// checks state that does not require nature
template <class SType>
bool is_action_correct(const SType& state, long stateid, const indvec& policy) {
    return state.is_action_correct(policy[stateid]);
}
} // namespace internal

} // namespace craam
