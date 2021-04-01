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

#define CRAAM_CONFIG_HPP // make sure to override the standard configuration

#include "rcraam_utils.hpp"

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <string>

/**
 * Converts a an integer vector to an Rcpp integer vector to pass it to
 * Rcpp DataFrame. Otherwise the vectors get converted to generic
 * numeric vectors.
 *
 * @param A vector of longs
 * @return An Rcpp vector of integers
 */
inline Rcpp::IntegerVector as_intvec(const craam::indvec& intvector) {
    return Rcpp::IntegerVector::import(intvector.cbegin(), intvector.cend());
}

/**
 * Constructs an R matrix from an Eigen matrix. All matrices are dense and the data is
 * copied. This method is not suitable for large matrices.
 */
inline Rcpp::NumericMatrix as_matrix(const Eigen::MatrixXd& matrix) {
    Rcpp::NumericMatrix result(matrix.rows(), matrix.cols());

    for (long i = 0; i < matrix.rows(); i++) {
        for (long j = 0; j < matrix.cols(); j++) {
            result(i, j) = matrix(i, j);
        }
    }
    return result;
}

/**
 * Constructs an R vector from an Eigen vector. May be slow.
 */
inline Rcpp::NumericVector as_vector(const Eigen::VectorXd& vector) {
    Rcpp::NumericVector result(vector.size());

    for (long i = 0; i < vector.size(); ++i) {
        result(i) = vector(i);
    }
    return result;
}

/**
 * Parses a data frame  to an MDP.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param frame Dataframe with columns: idstatefrom, idaction, idstateto, reward, probability.
 *              Multiple state-action-state rows have summed probabilities and averaged rewards.
 * @param force Whether transitions with probability 0 should be focibly added to the transitions.
 *              This makes a difference with robust MDPs.
 *
 * @returns Corresponding MDP definition
 */
inline craam::MDP mdp_from_dataframe(const Rcpp::DataFrame& data, bool force = false) {
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"], idaction = data["idaction"],
                        idstateto = data["idstateto"];
    Rcpp::NumericVector probability = data["probability"], reward = data["reward"];

    size_t n = data.nrow();
    craam::MDP m;

    for (size_t i = 0; i < n; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idstateto[i],
                              probability[i], reward[i], force);
    }
    return m;
}

/**
 * 
added by Monkie to avoid using range.
 *
 */
inline long bound_bisect(craam::indvec nums, long target, bool lower=true){
    long low = 0;
    long high = nums.size();
    while (low < high){
        long mid = (high + low)/2;
        if (nums[mid] > target || (lower && (nums[mid] == target))){
            high = mid;
        } else {
            low = mid+1;
        }
    }
    return low;
}

/**
 * Parses a MDPO dataframe  to an MDP.
 *
 * IMPORTANT: Assumes that the outcomes are sorted increasingly!
 *
 * Selects only one outcome from the dataframe which must have a column idoutcome.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param idstatefrom These are columns from the dataframe. The function takes processed inputs
 *                    in order to be more efficient.
 *
 * @param force Whether transitions with probability 0 should be focibly added to the transitions.
 *              This makes a difference with robust MDPs.
 * @param outcome_act Which outcome should be used
 *
 * @returns Corresponding MDP definition
 */
inline craam::MDP
mdp_from_mdpo_dataframe(const craam::indvec& idstatefrom, const craam::indvec idaction,
                        const craam::indvec idoutcome, const craam::indvec idstateto,
                        const craam::numvec probability, const craam::numvec reward,
                        long outcome_act, bool force) {
    // idstatefrom, idaction, idstateto, probability, reward
    const auto start = &idoutcome[bound_bisect(idoutcome, outcome_act,true)];
    const auto end = &idoutcome[bound_bisect(idoutcome, outcome_act,false)];

    const auto istart = std::distance(&idoutcome[0], start);
    const auto iend = std::distance(&idoutcome[0], end);

    craam::MDP m;
    for (size_t i = istart; i < iend; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idstateto[i],
                              probability[i], reward[i], force);
    }
    return m;
}
/**
 * Parses a data frame  to an MDPO. Each outcome represents a possible outcome of nature
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param frame Dataframe with columns: idstatefrom, idaction, idoutcome, idstateto, reward, probability.
 *              Multiple state-action-outcome-state rows have summed probabilities and averaged rewards.
 * @param force Whether transitions with probability 0 should be focibly added to the transitions.
 *              This makes a difference with robust MDPs.
 *
 * @returns Corresponding MDPO definition
 */
inline craam::MDPO mdpo_from_dataframe(const Rcpp::DataFrame& data, bool force = false) {
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"], idaction = data["idaction"],
                        idstateto = data["idstateto"], idoutcome = data["idoutcome"];
    Rcpp::NumericVector probability = data["probability"], reward = data["reward"];

    size_t n = data.nrow();
    craam::MDPO m;

    for (size_t i = 0; i < n; i++) {
        craam::add_transition(m, idstatefrom[i], idaction[i], idoutcome[i], idstateto[i],
                              probability[i], reward[i], force);
    }
    return m;
}

/**
 * Turns a dataframe `frame` to a matrix (array of arrays) of dimensions
 * dim1 x dim2. Index1 and index2 are the name of the columns with the
 * indices and value is the name of the value column. def_value is the
 * default value for any elements that are not provided.
 */
inline craam::numvecvec frame2matrix(const Rcpp::DataFrame& frame, size_t dim1,
                                     size_t dim2, const std::string& index1,
                                     const std::string& index2, const std::string& value,
                                     double def_value) {

    craam::numvecvec result(dim1);
    for (long i = 0; i < dim1; i++) {
        result[i] = craam::numvec(dim2, def_value);
    }

    Rcpp::IntegerVector idvec1 = frame[index1], idvec2 = frame[index2];
    Rcpp::NumericVector values = frame[value];

    for (long i = 0; i < idvec1.size(); i++) {
        long id1 = idvec1[i], id2 = idvec2[i];

        if (id1 < 0) Rcpp::stop("idstate must be non-negative");
        if (id1 > dim1)
            Rcpp::stop("idstate must be smaller than the number of MDP states");
        if (id2 < 0) Rcpp::stop("idaction must be non-negative");
        if (id2 > dim2)
            Rcpp::stop("idaction must be smaller than the number of actions for the "
                       "corresponding state");

        result[id1][id2] = values[i];
    }

    return result;
}

/**
 * Parses a data frame definition of values that correspond to states.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @tparam M Model that is either and MDP or MDPO
 * @tparam T Type of the value to parse
 *
 * @param mdp The definition of the MDP to know how many states and actions there are.
 * @param frame Dataframe with 2 comlumns, idstate, value. Here, idstate
 *              determines which value should be set.
 *              Only the last value is used if multiple rows are present.
 * @param def_value The default value for when frame does not
 *                  specify anything for the state action pair
 * @param value_column Name of the column with the value
 * @param param_name The name of the parameter which is being parsed.
 *                  This is used for error reporting.
 *
 * @tparam T Type of the value to parse
 *
 * @returns A vector over states with the included values
 */
template <class T>
inline std::vector<T> parse_s_values(std::size_t statecount, const Rcpp::DataFrame& frame,
                                     T def_value = 0,
                                     const std::string& value_column = "value",
                                     const std::string& param_name = "") {
    std::vector<T> result(statecount);
    Rcpp::IntegerVector idstates = frame["idstate"];
    Rcpp::NumericVector values = frame[value_column];

    for (std::size_t i = 0; i < idstates.size(); i++) {
        long idstate = idstates[i];

        if (idstate < 0) {
            Rcpp::stop(
                "idstate must be non-negative" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
        }

        if (idstate >= statecount) {
            Rcpp::stop(
                "idstate must be smaller than the number of MDP states" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
        }

        result[idstate] = values[i];
    }
    return result;
}

/**
 * Parses a data frame definition of values that correspond to states and
 * actions.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param mdp The definition of the MDP to know how many states and actions there are.
 * @param frame Dataframe with 3 comlumns, idstate, idaction, value. Here, idstate and idaction
 *              determine which value should be set.
 *              Only the last value is used if multiple rows are present.
 * @param def_value The default value for when frame does not specify anything for the state action pair
 * @param val_name Name of the value column
 * @param param_name The name of the parameter which is being parsed.
 *                  This is used for error reporting.
 *
 * @tparam M It is either and MDP or MDPO
 *
 * @returns A vector over states with an inner vector of actions
 */
template <typename M>
inline craam::numvecvec parse_sa_values(const M& mdp, const Rcpp::DataFrame& frame,
                                        double def_value = 0,
                                        const std::string& val_name = "value",
                                        const std::string& param_name = "") {

    craam::numvecvec result(mdp.size());
    for (long i = 0; i < mdp.size(); i++) {
        result[i] = craam::numvec(mdp[i].size(), def_value);
    }

    Rcpp::IntegerVector idstates = frame["idstate"], idactions = frame["idaction"];
    Rcpp::NumericVector values = frame[val_name];

    for (long i = 0; i < idstates.size(); i++) {
        long idstate = idstates[i], idaction = idactions[i];

        if (idstate < 0)
            Rcpp::stop(
                "idstate must be non-negative" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
        if (idstate >= mdp.size())
            Rcpp::stop(
                "idstate must be smaller than the number of MDP states" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
        if (idaction < 0)
            Rcpp::stop(
                "idaction must be non-negative" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
        if (idaction >= mdp[idstate].size())
            Rcpp::stop(
                "idaction must be smaller than the number of actions for the "
                "corresponding state" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));

        double value = values[i];
        result[idstate][idaction] = value;
    }

    return result;
}

/**
 * Parses a data frame definition of values that correspond to starting states, actions,
 * and taget states.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param mdp The definition of the MDP to know how many states and actions there are.
 * @param frame Dataframe with 3 comlumns, idstatefrom, idaction, idstateto, value.
 *              Here, idstate(from,to) and idaction determine which value should be set
 *              Only the last value is used if multiple rows are present.
 * @param def_value The default value for when frame does not specify anything for
 *              the state action pair
 * @param val_name Name of the value column
 * @param param_name The name of the parameter which is being parsed.
 *                  This is used for error reporting.
 *
 * @return A vector over states, action, with an inner vector of actions
 */
inline std::vector<craam::numvecvec>
parse_sas_values(const craam::MDP& mdp, const Rcpp::DataFrame& frame,
                 double def_value = 0, const std::string& val_name = "value",
                 const std::string& param_name = "") {

    std::vector<craam::numvecvec> result(mdp.size());
    for (long i = 0; i < mdp.size(); i++) {
        result[i] = craam::numvecvec(mdp[i].size());
        for (long j = 0; j < mdp[i].size(); j++) {
            // this is the number of non-zero transition probabilities
            result[i][j] = craam::numvec(mdp[i][j].size(), def_value);
        }
    }

    craam::indvec idstatesfrom = frame["idstatefrom"], idactions = frame["idaction"],
                  idstatesto = frame["idstateto"];
    craam::numvec values = frame[val_name];

    for (long i = 0; i < idstatesfrom.size(); i++) {
        long idstatefrom = idstatesfrom[i], idstateto = idstatesto[i],
             idaction = idactions[i];

        if (idstatefrom < 0) {
            Rcpp::warning(
                "idstatefrom must be non-negative" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
            continue;
        }
        if (idstatefrom >= mdp.size()) {
            Rcpp::warning(
                "idstatefrom must be smaller than the number of MDP states" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
            continue;
        }
        if (idaction < 0) {
            Rcpp::warning(
                "idaction must be non-negative" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
            continue;
        }
        if (idaction >= mdp[idstatefrom].size()) {
            Rcpp::warning(
                "idaction must be smaller than the number of actions for the "
                "corresponding state" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
            continue;
        }
        if (idstateto < 0) {
            Rcpp::warning(
                "idstateto must be non-negative" +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
            continue;
        }

        long indexto = mdp[idstatefrom][idaction].index_of(idstateto);
        //cout << idstatefrom << "," << idaction << "," << idstateto << "," << indexto
        //     << endl;

        if (indexto < 0) {
            Rcpp::warning(
                "idstateto must be one of the states with non-zero probability."
                "idstatefrom = " +
                std::to_string(idstatefrom) + ", idaction = " + std::to_string(idaction) +
                (param_name.empty() ? "." : " in parameter '" + param_name + "'."));
        } else {
            result[idstatefrom][idaction][indexto] = values[i];
        }
    }
    return result;
}

/**
 * Turns the definition of the sas nature output to a dataframe
 * @param mdp The definition of the mdp, needed to parse which transitions are possible
 *              from a given state and action
 * @param policy Policy to determine which actions are active
 * @param nature A vector over: state-from, action, state-to
 * @return
 */
inline Rcpp::DataFrame sanature_todataframe(const craam::MDP& mdp,
                                            const craam::indvec& policy,
                                            const craam::numvecvec& nature) {

    if (nature.size() != mdp.size())
        throw std::runtime_error("invalid number of states.");

    // construct output vectors
    craam::indvec out_statefrom, out_action, out_stateto;
    craam::numvec out_prob;

    // iterate over states
    for (size_t idstate = 0; idstate < mdp.size(); ++idstate) {
        const auto& state = mdp[idstate];

        long idaction = policy[idstate];

        // skip all actions in terminal states
        if (idaction < 0) continue;

        if (idaction >= state.size()) throw std::runtime_error("invalid policy");

        const auto& action = state[idaction];

        // check if the output states match
        if (action.size() != nature[idstate].size())
            throw std::runtime_error("invalid number of to-states");

        // iterate over all state tos
        for (size_t idstateto = 0; idstateto < action.size(); ++idstateto) {
            out_statefrom.push_back(idstate);
            out_action.push_back(idaction);
            out_stateto.push_back(idstateto);
            out_prob.push_back(nature[idstate][idstateto]);
        }
    }

    return Rcpp::DataFrame::create(Rcpp::_["idstatefrom"] = as_intvec(out_statefrom),
                                   Rcpp::_["idaction"] = as_intvec(out_action),
                                   Rcpp::_["idstateto"] = as_intvec(out_stateto),
                                   Rcpp::_["probability"] = out_prob);
}

/**
 * Turns the definition of the sas nature output to a dataframe
 * @param mdp The definition of the mdp, needed to parse which transitions are possible
 *              from a given state and action
 * @param policy Policy to determine which actions are active
 * @param nature A vector over: state-from, action, outcome
 * @return
 */
inline Rcpp::DataFrame sanature_out_todataframe(const craam::MDPO& mdpo,
                                                const craam::indvec& policy,
                                                const craam::numvecvec& nature) {

    if (nature.size() != mdpo.size())
        throw std::runtime_error("invalid number of states.");

    // construct output vectors
    craam::indvec out_statefrom, out_action, out_outcome;
    craam::numvec out_prob;

    // iterate over states
    for (size_t idstate = 0; idstate < mdpo.size(); ++idstate) {
        const auto& state = mdpo[idstate];

        long idaction = policy[idstate];

        // skip all actions in terminal states
        if (idaction < 0) continue;

        if (idaction >= state.size()) throw std::runtime_error("invalid policy");

        const auto& action = state[idaction];

        // check if the output states match
        if (action.size() != nature[idstate].size())
            throw std::runtime_error("invalid number of to-states");

        // iterate over all state tos
        for (size_t idoutcome = 0; idoutcome < action.size(); ++idoutcome) {
            out_statefrom.push_back(idstate);
            out_action.push_back(idaction);
            out_outcome.push_back(idoutcome);
            out_prob.push_back(nature[idstate][idoutcome]);
        }
    }

    return Rcpp::DataFrame::create(Rcpp::_["idstatefrom"] = as_intvec(out_statefrom),
                                   Rcpp::_["idaction"] = as_intvec(out_action),
                                   Rcpp::_["idoutcome"] = as_intvec(out_outcome),
                                   Rcpp::_["probability"] = out_prob);
}

/**
 * Turns the definition of the sas nature output to a dataframe
 *
 * @param mdp The definition of the mdp, needed to parse which transitions are possible
 *              from a given state and action
 * @param nature A vector over: state-from, action, outcome
 *
 * @return DataFrame with outcome weights for each states
 */
inline Rcpp::DataFrame output_snature(const craam::MDPO& mdpo,
                                      const craam::numvecvec& nature) {

    if (nature.size() != mdpo.size())
        throw std::runtime_error("invalid number of states.");

    // construct output vectors
    craam::indvec out_statefrom, out_outcome;
    craam::numvec out_prob;

    // iterate over states
    for (size_t idstate = 0; idstate < mdpo.size(); ++idstate) {
        const auto& nature_state = nature[idstate];
        // skip terminal states that have no outcomes
        if (nature_state.empty()) continue;

        // iterate over all state tos
        for (size_t idoutcome = 0; idoutcome < nature_state.size(); ++idoutcome) {
            out_statefrom.push_back(idstate);
            out_outcome.push_back(idoutcome);
            out_prob.push_back(nature[idstate][idoutcome]);
        }
    }

    return Rcpp::DataFrame::create(Rcpp::_["idstatefrom"] = as_intvec(out_statefrom),
                                   Rcpp::_["idoutcome"] = as_intvec(out_outcome),
                                   Rcpp::_["probability"] = out_prob);
}

/**
 * Turns the definition of the sas nature output to a dataframe
 * @param mdp The definition of the mdp, needed to parse which transitions are possible
 *              from a given state and action
 * @param nature A vector over: state-from, action, state-to
 * @return
 */
inline Rcpp::DataFrame
sasnature_todataframe(const craam::MDP& mdp,
                      const std::vector<craam::numvecvec>& nature) {

    if (nature.size() != mdp.size())
        throw std::runtime_error("invalid number of states.");

    // construct output vectors
    craam::indvec out_statefrom, out_action, out_stateto;
    craam::numvec out_prob;

    // iterate over states
    for (size_t idstate = 0; idstate < mdp.size(); ++idstate) {
        const auto& state = mdp[idstate];

        if (state.size() != nature[idstate].size())
            throw std::runtime_error("invalid number of actions.");

        for (size_t idaction = 0; idaction < state.size(); ++idaction) {
            const auto& action = state[idaction];

            // check if the output states match
            if (action.size() != nature[idstate][idaction].size())
                throw std::runtime_error("invalid number of to-states");

            // iterate over all state tos
            for (size_t idstateto = 0; idstateto < action.size(); ++idstateto) {
                out_statefrom.push_back(idstate);
                out_action.push_back(idaction);
                out_stateto.push_back(idstateto);
                out_prob.push_back(nature[idstate][idaction][idstateto]);
            }
        }
    }

    return Rcpp::DataFrame::create(Rcpp::_["idstatefrom"] = as_intvec(out_statefrom),
                                   Rcpp::_["idaction"] = as_intvec(out_action),
                                   Rcpp::_["idstateto"] = as_intvec(out_stateto),
                                   Rcpp::_["probability"] = out_prob);
}

/**
 * Maps the output from craam::pack_actions to a datafram
 * @param actionmap Output from pack_actions (a list for each state of old action indices)
 * @return Dataframe with idstate, idaction_old, idaction_new where _old is before packing
 *  and _new is after packing
 */
inline Rcpp::DataFrame actionmap2df(const std::vector<craam::indvec>& actionmap) {
    craam::indvec idstate, idaction_old, idaction_new;

    for (std::size_t istate = 0; istate < actionmap.size(); ++istate) {
        const auto& state_alist = actionmap[istate];
        for (std::size_t iaction_new; iaction_new < state_alist.size(); ++iaction_new) {
            idstate.push_back(istate);
            idaction_new.push_back(iaction_new);
            idaction_old.push_back(actionmap[istate][iaction_new]);
        }
    }

    return Rcpp::DataFrame::create(Rcpp::_["idstate"] = as_intvec(idstate),
                                   Rcpp::_["idaction_old"] = as_intvec(idaction_old),
                                   Rcpp::_["idaction_new"] = as_intvec(idaction_new));
}
