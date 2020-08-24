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

// skip configuration if it is already provided
#ifndef CRAAM_CONFIG_HPP
#include "config.hpp"
#endif // CRAAM_CONFIG_HPP

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace craam {

/// Default precision type
using prec_t = double;
/// Numerical vector
using numvec = std::vector<prec_t>;
/// Vector of indices
using indvec = std::vector<long>;
/// Vector of size_t
using sizvec = std::vector<size_t>;
/// A vector of numeric vectors
using numvecvec = std::vector<numvec>;
/// A probability distribution
using pdist = numvec;

/** Probability list */
using prob_list_t = std::vector<prec_t>;

/** Probability matrix */
using prob_matrix_t = std::vector<prob_list_t>;

/** Pair of a vector and a scalar */
using vec_scal_t = std::pair<numvec, prec_t>;
/// Tuple of a index, vector and a scalar
using ind_vec_scal_t = std::tuple<prec_t, numvec, prec_t>;

/** Default solution precision */
constexpr prec_t SOLPREC = 0.0001;

/** Default solution precision */
constexpr prec_t EPSILON = 1e-6;

/** Default number of iterations */
constexpr unsigned long MAXITER = 100000;

/// Numerical threshold for reporting errors
constexpr prec_t THRESHOLD = 1e-6;

/// An exception class that indicates that the
/// provided model was misspecified
class ModelError : public std::exception {
public:
    std::string message;
    long idstate;
    long idaction;
    long idoutcome;
    std::string fullmessage;

    /// updates the full message if anything changes
    void update_fullmsg() {
        fullmessage = "State: " + std::to_string(idstate) +
                      ", Action: " + std::to_string(idaction) +
                      (idoutcome >= 0 ? ", Outcome: " + std::to_string(idoutcome) : "") +
                      ", Message: '" + message + "'.";
    }

public:
    ModelError(std::string message, long idstate = -1, long idaction = -1,
               long idoutcome = -1)
        : message(message), idstate(idstate), idaction(idaction), idoutcome(idoutcome) {
        update_fullmsg();
    };

    /// Sets the state information
    void set_state(long idstate) {
        this->idstate = idstate;
        update_fullmsg();
    }

    /// Sets the action information
    void set_action(long idaction) {
        this->idaction = idaction;
        update_fullmsg();
    }

    /// Sets the outcome information
    void set_outcome(long idoutcome) {
        this->idoutcome = idoutcome;
        update_fullmsg();
    }

    /**
     * Returns the content of the exception. The caller takes
     * the ownership of the string
     */
    const char* what() const noexcept override { return fullmessage.c_str(); };
};

/** This is a useful print functionality for debugging.  */
template <class T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    for (const auto& p : vec) {
        os << p << " ";
    }
    return os;
}

/// Vector to string
template <class T> inline std::string to_string(const std::vector<T>& v) {

    std::stringstream ss;

    ss << "[";
    for (const auto& x : v) {
        ss << x << ",";
    }
    // replace the last comma
    ss.seekp(long(ss.tellp()) - 1l);
    ss << "]";
    return ss.str();
}

/**
 * Sort indices by values in ascending order
 *
 * @param v List of values
 * @return Sorted indices
 */
template <typename T> inline sizvec sort_indexes(std::vector<T> const& v) {
    // initialize original index locations
    sizvec idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

/**
 * Sort indices by values in descending order
 *
 * @param v List of values
 * @return Sorted indices
 */
template <typename T> inline sizvec sort_indexes_desc(std::vector<T> const& v) {
    // initialize original index locations
    sizvec idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

    return idx;
}

/**
 * Computes the l1 norm between two vectors of equal length
 */
inline prec_t l1norm(numvec p1, numvec p2) {
    prec_t result = 0;
    for (size_t i = 0; i < p1.size(); i++)
        result += std::abs(p1[i] - p2[i]);
    return result;
}

/**
 *  Generates linearly spaced points
 *
 * @param a Start value
 * @param b End value
 * @param N Number of points
 */
template <typename T = double> inline std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N - 1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

/// Zips two vectors
template <class T1, class T2>
inline std::vector<std::pair<T1, T2>> zip(const std::vector<T1>& v1,
                                          const std::vector<T2>& v2) {

    assert(v1.size() == v2.size());
    std::vector<std::pair<T1, T2>> result(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        result[i] = make_pair(v1[i], v2[i]);
    }
    return result;
}

/// Zips two vectors of vectors
template <class T1, class T2>
inline std::vector<std::vector<std::pair<T1, T2>>>
zip(const std::vector<std::vector<T1>>& v1, const std::vector<std::vector<T2>>& v2) {

    assert(v1.size() == v2.size());
    std::vector<std::vector<std::pair<T1, T2>>> result(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        result[i] = zip(v1[i], v2[i]);
    }
    return result;
}

/// Zips a single value with a vector
template <class T1, class T2>
inline std::vector<std::pair<T1, T2>> zip(const T1& v1, const std::vector<T2>& v2) {

    std::vector<std::pair<T1, T2>> result(v2.size());
    for (size_t i = 0; i < v2.size(); i++) {
        result[i] = make_pair(v1, v2[i]);
    }
    return result;
}

/// Zips a single value with a vector of vectors
template <class T1, class T2>
inline std::vector<std::vector<std::pair<T1, T2>>>
zip(const T1& v1, const std::vector<std::vector<T2>>& v2) {

    std::vector<std::vector<std::pair<T1, T2>>> result(v2.size());
    for (size_t i = 0; i < v2.size(); i++) {
        result[i] = zip(v1, v2[i]);
    }
    return result;
}

/**
 * Takes a vector of pairs and constructs two vectors from each
 * component of the pair
 */
template <class T1, class T2>
inline std::pair<std::vector<T1>, std::vector<T2>>
unzip(const std::vector<std::pair<T1, T2>>& values) {
    std::vector<T1> first;
    first.reserve(values.size());
    std::vector<T2> second;
    second.reserve(values.size());
    for (const auto& x : values) {
        first.push_back(x.first);
        second.push_back(x.second);
    }
    return {first, second};
}

/**
 * Checks whether the iterator is a probability distribution. That means
 * it checks whether each element is non-negative and that the elements
 * sum to (approximately) 1 (the tolerance is epsilon.
 */
template <class ForwardIterator>
inline bool is_probability_dist(ForwardIterator first, ForwardIterator last) {
    if (first == last) return false;

    prec_t sum = *first;
    while (++first != last) {
        if (*first < 0) return false;
        sum += *first;
    }
    return std::abs(sum - 1.0) < EPSILON;
}

// implement clamp when not provided by the library (in pre c++17 code)
#ifndef __cpp_lib_clamp
template <class T, class Compare>
inline constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp) {
    return assert(!comp(hi, lo)), comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

template <class T> inline constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
    return craam::clamp<T, std::less<>>(v, lo, hi, std::less<>());
}
#endif

/**
 * Multiplies the vector by a value and returns it
 */
inline numvec multiply(numvec vct, prec_t value) {
    // intentionally passed by value (to enable a move and handling rvalues)
    for (int i = 0; i < int(vct.size()); ++i)
        vct[i] *= value;
    return vct;
}

/**
 * Multiplies the vector by a value inplace
 */
inline void multiply_inplace(numvec& vct, prec_t value) {
    for (int i = 0; i < int(vct.size()); ++i)
        vct[i] *= value;
}

class ProbDst {
protected:
    numvec content;

public:
    template <class... Args>
    ProbDst(Args&&... args) : content(std::forward<Args>(args)...) {
        // check whether the values sum to 1 and are non-negative
        if (!content.empty()) {
            prec_t sum = 0.0;
            for (prec_t n : content) {
                if (n < 0)
                    throw std::invalid_argument(
                        "Probability elements must be non-negative.");
                sum += n;
            }
            if (std::abs(sum - 1.0) > EPSILON)
                throw std::invalid_argument("Probability distribution must sum to 1.");
        }
    }

    const numvec& vector() const { return content; }

    size_t size() const { return content.size(); }
    const prec_t& operator[](size_t index) const { return content[index]; }
    bool empty() const { return content.empty(); }
    auto cbegin() const { return content.cbegin(); }
    auto cend() const { return content.cend(); }
};

namespace internal {
/// Reports the exception that cannot be passed up from an openmp block
/// @param e The exception caught
/// @param fname Name of the offending function
inline void openmp_exception_handler(const std::exception& e, const std::string& fname) {
    std::cerr << "********" << std::endl
              << "Caught an exception in the OPENMP block: " << std::endl
              << e.what() << std::endl
              << "in method " << fname << std::endl
              << "********" << std::endl;
}

} // namespace internal
} // namespace craam
