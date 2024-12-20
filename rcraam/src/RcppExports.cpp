// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// worstcase_l1
Rcpp::List worstcase_l1(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst, double budget);
RcppExport SEXP _rcraam_worstcase_l1(SEXP valueSEXP, SEXP reference_dstSEXP, SEXP budgetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value(valueSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reference_dst(reference_dstSEXP);
    Rcpp::traits::input_parameter< double >::type budget(budgetSEXP);
    rcpp_result_gen = Rcpp::wrap(worstcase_l1(value, reference_dst, budget));
    return rcpp_result_gen;
END_RCPP
}
// worstcase_l1_w
Rcpp::List worstcase_l1_w(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst, Rcpp::NumericVector w, double budget);
RcppExport SEXP _rcraam_worstcase_l1_w(SEXP valueSEXP, SEXP reference_dstSEXP, SEXP wSEXP, SEXP budgetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value(valueSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reference_dst(reference_dstSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type budget(budgetSEXP);
    rcpp_result_gen = Rcpp::wrap(worstcase_l1_w(value, reference_dst, w, budget));
    return rcpp_result_gen;
END_RCPP
}
// worstcase_l1_w_gurobi
Rcpp::List worstcase_l1_w_gurobi(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst, Rcpp::NumericVector w, double budget);
RcppExport SEXP _rcraam_worstcase_l1_w_gurobi(SEXP valueSEXP, SEXP reference_dstSEXP, SEXP wSEXP, SEXP budgetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value(valueSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reference_dst(reference_dstSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type budget(budgetSEXP);
    rcpp_result_gen = Rcpp::wrap(worstcase_l1_w_gurobi(value, reference_dst, w, budget));
    return rcpp_result_gen;
END_RCPP
}
// worstcase_linf_w_gurobi
Rcpp::List worstcase_linf_w_gurobi(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst, Rcpp::NumericVector w, double budget);
RcppExport SEXP _rcraam_worstcase_linf_w_gurobi(SEXP valueSEXP, SEXP reference_dstSEXP, SEXP wSEXP, SEXP budgetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value(valueSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reference_dst(reference_dstSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type budget(budgetSEXP);
    rcpp_result_gen = Rcpp::wrap(worstcase_linf_w_gurobi(value, reference_dst, w, budget));
    return rcpp_result_gen;
END_RCPP
}
// avar
Rcpp::List avar(Rcpp::NumericVector value, Rcpp::NumericVector reference_dst, double alpha);
RcppExport SEXP _rcraam_avar(SEXP valueSEXP, SEXP reference_dstSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value(valueSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reference_dst(reference_dstSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(avar(value, reference_dst, alpha));
    return rcpp_result_gen;
END_RCPP
}
// pack_actions
Rcpp::List pack_actions(Rcpp::DataFrame mdp);
RcppExport SEXP _rcraam_pack_actions(SEXP mdpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    rcpp_result_gen = Rcpp::wrap(pack_actions(mdp));
    return rcpp_result_gen;
END_RCPP
}
// mdp_clean
Rcpp::DataFrame mdp_clean(Rcpp::DataFrame mdp);
RcppExport SEXP _rcraam_mdp_clean(SEXP mdpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_clean(mdp));
    return rcpp_result_gen;
END_RCPP
}
// solve_mdp
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, double discount, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, Rcpp::Nullable<Rcpp::DataFrame> value_init, bool pack_actions, int show_progress);
RcppExport SEXP _rcraam_solve_mdp(SEXP mdpSEXP, SEXP discountSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP value_initSEXP, SEXP pack_actionsSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type value_init(value_initSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< int >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(solve_mdp(mdp, discount, algorithm, policy_fixed, maxresidual, iterations, timeout, value_init, pack_actions, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// solve_mdp_rand
Rcpp::List solve_mdp_rand(Rcpp::DataFrame mdp, double discount, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, Rcpp::Nullable<Rcpp::DataFrame> value_init, int show_progress);
RcppExport SEXP _rcraam_solve_mdp_rand(SEXP mdpSEXP, SEXP discountSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP value_initSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type value_init(value_initSEXP);
    Rcpp::traits::input_parameter< int >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(solve_mdp_rand(mdp, discount, algorithm, policy_fixed, maxresidual, iterations, timeout, value_init, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// compute_qvalues
Rcpp::DataFrame compute_qvalues(Rcpp::DataFrame mdp, double discount, Rcpp::DataFrame valuefunction);
RcppExport SEXP _rcraam_compute_qvalues(SEXP mdpSEXP, SEXP discountSEXP, SEXP valuefunctionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type valuefunction(valuefunctionSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_qvalues(mdp, discount, valuefunction));
    return rcpp_result_gen;
END_RCPP
}
// rsolve_mdp_sa
Rcpp::List rsolve_mdp_sa(Rcpp::DataFrame mdp, double discount, Rcpp::String nature, SEXP nature_par, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, Rcpp::Nullable<Rcpp::DataFrame> value_init, bool pack_actions, bool output_tran, int show_progress);
RcppExport SEXP _rcraam_rsolve_mdp_sa(SEXP mdpSEXP, SEXP discountSEXP, SEXP natureSEXP, SEXP nature_parSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP value_initSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type nature(natureSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nature_par(nature_parSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type value_init(value_initSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< int >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(rsolve_mdp_sa(mdp, discount, nature, nature_par, algorithm, policy_fixed, maxresidual, iterations, timeout, value_init, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// rsolve_mdpo_sa
Rcpp::List rsolve_mdpo_sa(Rcpp::DataFrame mdpo, double discount, Rcpp::String nature, SEXP nature_par, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, Rcpp::Nullable<Rcpp::DataFrame> value_init, bool pack_actions, bool output_tran, int show_progress);
RcppExport SEXP _rcraam_rsolve_mdpo_sa(SEXP mdpoSEXP, SEXP discountSEXP, SEXP natureSEXP, SEXP nature_parSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP value_initSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdpo(mdpoSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type nature(natureSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nature_par(nature_parSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type value_init(value_initSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< int >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(rsolve_mdpo_sa(mdpo, discount, nature, nature_par, algorithm, policy_fixed, maxresidual, iterations, timeout, value_init, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// srsolve_mdpo
Rcpp::List srsolve_mdpo(Rcpp::DataFrame mdpo, Rcpp::DataFrame init_distribution, double discount, double alpha, double beta, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> model_distribution, Rcpp::String output_filename);
RcppExport SEXP _rcraam_srsolve_mdpo(SEXP mdpoSEXP, SEXP init_distributionSEXP, SEXP discountSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP algorithmSEXP, SEXP model_distributionSEXP, SEXP output_filenameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdpo(mdpoSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type init_distribution(init_distributionSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type model_distribution(model_distributionSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type output_filename(output_filenameSEXP);
    rcpp_result_gen = Rcpp::wrap(srsolve_mdpo(mdpo, init_distribution, discount, alpha, beta, algorithm, model_distribution, output_filename));
    return rcpp_result_gen;
END_RCPP
}
// rsolve_mdp_s
Rcpp::List rsolve_mdp_s(Rcpp::DataFrame mdp, double discount, Rcpp::String nature, SEXP nature_par, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, Rcpp::Nullable<Rcpp::DataFrame> value_init, bool pack_actions, bool output_tran, int show_progress);
RcppExport SEXP _rcraam_rsolve_mdp_s(SEXP mdpSEXP, SEXP discountSEXP, SEXP natureSEXP, SEXP nature_parSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP value_initSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type nature(natureSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nature_par(nature_parSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type value_init(value_initSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< int >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(rsolve_mdp_s(mdp, discount, nature, nature_par, algorithm, policy_fixed, maxresidual, iterations, timeout, value_init, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// rsolve_mdpo_s
Rcpp::List rsolve_mdpo_s(Rcpp::DataFrame mdpo, double discount, Rcpp::String nature, SEXP nature_par, Rcpp::String algorithm, Rcpp::Nullable<Rcpp::DataFrame> policy_fixed, double maxresidual, size_t iterations, double timeout, Rcpp::Nullable<Rcpp::DataFrame> value_init, bool pack_actions, bool output_tran, int show_progress);
RcppExport SEXP _rcraam_rsolve_mdpo_s(SEXP mdpoSEXP, SEXP discountSEXP, SEXP natureSEXP, SEXP nature_parSEXP, SEXP algorithmSEXP, SEXP policy_fixedSEXP, SEXP maxresidualSEXP, SEXP iterationsSEXP, SEXP timeoutSEXP, SEXP value_initSEXP, SEXP pack_actionsSEXP, SEXP output_tranSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdpo(mdpoSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type nature(natureSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nature_par(nature_parSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_fixed(policy_fixedSEXP);
    Rcpp::traits::input_parameter< double >::type maxresidual(maxresidualSEXP);
    Rcpp::traits::input_parameter< size_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type timeout(timeoutSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type value_init(value_initSEXP);
    Rcpp::traits::input_parameter< bool >::type pack_actions(pack_actionsSEXP);
    Rcpp::traits::input_parameter< bool >::type output_tran(output_tranSEXP);
    Rcpp::traits::input_parameter< int >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(rsolve_mdpo_s(mdpo, discount, nature, nature_par, algorithm, policy_fixed, maxresidual, iterations, timeout, value_init, pack_actions, output_tran, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// revaluate_mdpo_rnd
Rcpp::DataFrame revaluate_mdpo_rnd(Rcpp::DataFrame mdpo, double discount, Rcpp::Nullable<Rcpp::DataFrame> policy_rnd, Rcpp::Nullable<Rcpp::DataFrame> initial, bool show_progress);
RcppExport SEXP _rcraam_revaluate_mdpo_rnd(SEXP mdpoSEXP, SEXP discountSEXP, SEXP policy_rndSEXP, SEXP initialSEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdpo(mdpoSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type policy_rnd(policy_rndSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::DataFrame> >::type initial(initialSEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress(show_progressSEXP);
    rcpp_result_gen = Rcpp::wrap(revaluate_mdpo_rnd(mdpo, discount, policy_rnd, initial, show_progress));
    return rcpp_result_gen;
END_RCPP
}
// rcraam_set_threads
void rcraam_set_threads(int n);
RcppExport SEXP _rcraam_rcraam_set_threads(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcraam_set_threads(n);
    return R_NilValue;
END_RCPP
}
// gurobi_set_param
void gurobi_set_param(Rcpp::String optimizer, Rcpp::String param, Rcpp::String value);
RcppExport SEXP _rcraam_gurobi_set_param(SEXP optimizerSEXP, SEXP paramSEXP, SEXP valueSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String >::type optimizer(optimizerSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type param(paramSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type value(valueSEXP);
    gurobi_set_param(optimizer, param, value);
    return R_NilValue;
END_RCPP
}
// mdp_from_samples
Rcpp::DataFrame mdp_from_samples(Rcpp::DataFrame samples_frame);
RcppExport SEXP _rcraam_mdp_from_samples(SEXP samples_frameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type samples_frame(samples_frameSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_from_samples(samples_frame));
    return rcpp_result_gen;
END_RCPP
}
// matrix_mdp_lp
Rcpp::List matrix_mdp_lp(Rcpp::DataFrame mdp, double discount);
RcppExport SEXP _rcraam_matrix_mdp_lp(SEXP mdpSEXP, SEXP discountSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< double >::type discount(discountSEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_mdp_lp(mdp, discount));
    return rcpp_result_gen;
END_RCPP
}
// matrix_mdp_transition
Rcpp::List matrix_mdp_transition(Rcpp::DataFrame mdp, Rcpp::DataFrame policy);
RcppExport SEXP _rcraam_matrix_mdp_transition(SEXP mdpSEXP, SEXP policySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type policy(policySEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_mdp_transition(mdp, policy));
    return rcpp_result_gen;
END_RCPP
}
// rcraam_supports_gurobi
bool rcraam_supports_gurobi();
RcppExport SEXP _rcraam_rcraam_supports_gurobi() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcraam_supports_gurobi());
    return rcpp_result_gen;
END_RCPP
}
// mdp_example
Rcpp::DataFrame mdp_example(Rcpp::String name);
RcppExport SEXP _rcraam_mdp_example(SEXP nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String >::type name(nameSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_example(name));
    return rcpp_result_gen;
END_RCPP
}
// mdp_inventory
Rcpp::DataFrame mdp_inventory(Rcpp::List params);
RcppExport SEXP _rcraam_mdp_inventory(SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_inventory(params));
    return rcpp_result_gen;
END_RCPP
}
// mdp_population
Rcpp::DataFrame mdp_population(int capacity, int initial, Rcpp::NumericMatrix growth_rates_exp, Rcpp::NumericMatrix growth_rates_std, Rcpp::NumericMatrix rewards, double external_mean, double external_std, Rcpp::String s_growth_model);
RcppExport SEXP _rcraam_mdp_population(SEXP capacitySEXP, SEXP initialSEXP, SEXP growth_rates_expSEXP, SEXP growth_rates_stdSEXP, SEXP rewardsSEXP, SEXP external_meanSEXP, SEXP external_stdSEXP, SEXP s_growth_modelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type capacity(capacitySEXP);
    Rcpp::traits::input_parameter< int >::type initial(initialSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type growth_rates_exp(growth_rates_expSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type growth_rates_std(growth_rates_stdSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type rewards(rewardsSEXP);
    Rcpp::traits::input_parameter< double >::type external_mean(external_meanSEXP);
    Rcpp::traits::input_parameter< double >::type external_std(external_stdSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type s_growth_model(s_growth_modelSEXP);
    rcpp_result_gen = Rcpp::wrap(mdp_population(capacity, initial, growth_rates_exp, growth_rates_std, rewards, external_mean, external_std, s_growth_model));
    return rcpp_result_gen;
END_RCPP
}
// simulate_mdp
Rcpp::DataFrame simulate_mdp(Rcpp::DataFrame mdp, int initial_state, Rcpp::DataFrame policy, int horizon, int episodes, Rcpp::Nullable<unsigned int> seed);
RcppExport SEXP _rcraam_simulate_mdp(SEXP mdpSEXP, SEXP initial_stateSEXP, SEXP policySEXP, SEXP horizonSEXP, SEXP episodesSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type mdp(mdpSEXP);
    Rcpp::traits::input_parameter< int >::type initial_state(initial_stateSEXP);
    Rcpp::traits::input_parameter< Rcpp::DataFrame >::type policy(policySEXP);
    Rcpp::traits::input_parameter< int >::type horizon(horizonSEXP);
    Rcpp::traits::input_parameter< int >::type episodes(episodesSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<unsigned int> >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(simulate_mdp(mdp, initial_state, policy, horizon, episodes, seed));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rcraam_worstcase_l1", (DL_FUNC) &_rcraam_worstcase_l1, 3},
    {"_rcraam_worstcase_l1_w", (DL_FUNC) &_rcraam_worstcase_l1_w, 4},
    {"_rcraam_worstcase_l1_w_gurobi", (DL_FUNC) &_rcraam_worstcase_l1_w_gurobi, 4},
    {"_rcraam_worstcase_linf_w_gurobi", (DL_FUNC) &_rcraam_worstcase_linf_w_gurobi, 4},
    {"_rcraam_avar", (DL_FUNC) &_rcraam_avar, 3},
    {"_rcraam_pack_actions", (DL_FUNC) &_rcraam_pack_actions, 1},
    {"_rcraam_mdp_clean", (DL_FUNC) &_rcraam_mdp_clean, 1},
    {"_rcraam_solve_mdp", (DL_FUNC) &_rcraam_solve_mdp, 10},
    {"_rcraam_solve_mdp_rand", (DL_FUNC) &_rcraam_solve_mdp_rand, 9},
    {"_rcraam_compute_qvalues", (DL_FUNC) &_rcraam_compute_qvalues, 3},
    {"_rcraam_rsolve_mdp_sa", (DL_FUNC) &_rcraam_rsolve_mdp_sa, 13},
    {"_rcraam_rsolve_mdpo_sa", (DL_FUNC) &_rcraam_rsolve_mdpo_sa, 13},
    {"_rcraam_srsolve_mdpo", (DL_FUNC) &_rcraam_srsolve_mdpo, 8},
    {"_rcraam_rsolve_mdp_s", (DL_FUNC) &_rcraam_rsolve_mdp_s, 13},
    {"_rcraam_rsolve_mdpo_s", (DL_FUNC) &_rcraam_rsolve_mdpo_s, 13},
    {"_rcraam_revaluate_mdpo_rnd", (DL_FUNC) &_rcraam_revaluate_mdpo_rnd, 5},
    {"_rcraam_rcraam_set_threads", (DL_FUNC) &_rcraam_rcraam_set_threads, 1},
    {"_rcraam_gurobi_set_param", (DL_FUNC) &_rcraam_gurobi_set_param, 3},
    {"_rcraam_mdp_from_samples", (DL_FUNC) &_rcraam_mdp_from_samples, 1},
    {"_rcraam_matrix_mdp_lp", (DL_FUNC) &_rcraam_matrix_mdp_lp, 2},
    {"_rcraam_matrix_mdp_transition", (DL_FUNC) &_rcraam_matrix_mdp_transition, 2},
    {"_rcraam_rcraam_supports_gurobi", (DL_FUNC) &_rcraam_rcraam_supports_gurobi, 0},
    {"_rcraam_mdp_example", (DL_FUNC) &_rcraam_mdp_example, 1},
    {"_rcraam_mdp_inventory", (DL_FUNC) &_rcraam_mdp_inventory, 1},
    {"_rcraam_mdp_population", (DL_FUNC) &_rcraam_mdp_population, 8},
    {"_rcraam_simulate_mdp", (DL_FUNC) &_rcraam_simulate_mdp, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_rcraam(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
