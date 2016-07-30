#define CXX_COMPILE 

#ifdef _MSC_VER
#include <windows.h>
#endif

extern "C" {
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "up/up.h"
#include "up/flags.h"
#include "bprolog.h"
#include "core/random.h"
#include "core/gamma.h"
#include "up/graph.h"
#include "up/util.h"
#include "up/em.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/viterbi.h"
#include "up/graph_aux.h"
#include "up/nonlinear_eq.h"
#include "up/scc.h"
#include "up/rank.h"
#include "up/crf.h"
#include "up/crf_learn.h"
#include "up/crf_learn_aux.h"
}

#include "eigen/Core"
#include "eigen/LU"

#include <iostream>
#include <set>
#include <cmath>


using namespace Eigen;


int run_rank_learn(struct EM_Engine* em_ptr) {

	printf("this is test code!!\n");
	return 1;
	int	 r, iterate, old_valid, converged, saved = 0;
	double  likelihood, log_prior=0;
	double  lambda, old_lambda = 0.0;

	//config_em(em_ptr);
	double start_time=getCPUTime();
	init_scc();
	double scc_time=getCPUTime();
	//start EM
	double itemp = 1.0;
	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#cyc-em-iters", r);
		initialize_params();
		iterate = 0;
		while (1) {
			old_valid = 0;
			while (1) {
				if (CTRLC_PRESSED) {
					SHOW_PROGRESS_INTR();
					RET_ERR(err_ctrl_c_pressed);
				}

				
				//RET_ON_ERR(em_ptr->compute_inside());
				compute_inside_linear();
				//RET_ON_ERR(em_ptr->examine_inside());
				//examine_inside_linear_cycle();
				//likelihood = em_ptr->compute_likelihood();
				likelihood=compute_likelihood_scaling_none();
				log_prior  = em_ptr->smooth ? em_ptr->compute_log_prior() : 0.0;
				lambda = likelihood + log_prior;
				if (verb_em) {
					if (em_ptr->smooth) {
						prism_printf("iteration #%d:\tlog_likelihood=%.9f\tlog_prior=%.9f\tlog_post=%.9f\n", iterate, likelihood, log_prior, lambda);
					}else {
						prism_printf("iteration #%d:\tlog_likelihood=%.9f\n", iterate, likelihood);
						if(scc_debug_level>=4) {
							print_eq();
						}
					}
				}

				if (!std::isfinite(lambda)) {
					emit_internal_error("invalid log likelihood or log post: %s (at iteration #%d)",
							std::isnan(lambda) ? "NaN" : "infinity", iterate);
					RET_ERR(ierr_invalid_likelihood);
				}
				if (old_valid && old_lambda - lambda > prism_epsilon) {
					emit_error("log likelihood or log post decreased [old: %.9f, new: %.9f] (at iteration #%d)",
							old_lambda, lambda, iterate);
					RET_ERR(err_invalid_likelihood);
				}

				converged = (old_valid && lambda - old_lambda <= prism_epsilon);
				if (converged || REACHED_MAX_ITERATE(iterate)) {
					break;
				}

				old_lambda = lambda;
				old_valid  = 1;

				//RET_ON_ERR(em_ptr->compute_expectation());
				compute_expectation_linear();

				SHOW_PROGRESS(iterate);
				RET_ON_ERR(em_ptr->update_params());
				//update_params();
				iterate++;
			}

			/* [21 Aug 2007, by yuizumi]
			 * Note that 1.0 can be represented exactly in IEEE 754.
			 */
			if (itemp == 1.0) {
				break;
			}
			itemp *= itemp_rate;
			if (itemp >= 1.0) {
				itemp = 1.0;
			}
	
		}

		SHOW_PROGRESS_TAIL(converged, iterate, lambda);

		if (r == 0 || lambda > em_ptr->lambda) {
			em_ptr->lambda     = lambda;
			em_ptr->likelihood = likelihood;
			em_ptr->iterate    = iterate;

			saved = (r < num_restart - 1);
			if (saved) {
				save_params();
			}
		}
	}
	if (saved) {
		restore_params();
	}
	//END EM

	if(scc_debug_level>=1) {
		print_sccs_statistics();
	}
	double solution_time=getCPUTime();
	//free data
	free_scc();
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}

	em_ptr->bic = compute_bic(em_ptr->likelihood);
	em_ptr->cs  = em_ptr->smooth ? compute_cs(em_ptr->likelihood) : 0.0;
	return BP_TRUE;
}

void config_rank_learn(EM_ENG_PTR em_ptr) {
	if (log_scale) {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_log_exp : compute_inside_scaling_log_exp;
		em_ptr->examine_inside      = examine_inside_scaling_log_exp;
		em_ptr->compute_expectation = compute_expectation_scaling_log_exp;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_log_exp;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = em_ptr->smooth ? update_params_smooth : update_params;
	} else {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_none : compute_inside_scaling_none;
		em_ptr->examine_inside      = examine_inside_scaling_none;
		em_ptr->compute_expectation = compute_expectation_scaling_none;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_none;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = em_ptr->smooth ? update_params_smooth : update_params;
	}
}

extern "C"
int pc_rank_learn_7(void) {
	struct EM_Engine em_eng;
	RET_ON_ERR(check_smooth(&em_eng.smooth));
	config_rank_learn(&em_eng);
	//scc_debug_level = bpx_get_integer(bpx_get_call_arg(7,7));
	run_rank_learn(&em_eng);
	return
	    bpx_unify(bpx_get_call_arg(1,7), bpx_build_integer(em_eng.iterate   )) &&
	    bpx_unify(bpx_get_call_arg(2,7), bpx_build_float  (em_eng.lambda    )) &&
	    bpx_unify(bpx_get_call_arg(3,7), bpx_build_float  (em_eng.likelihood)) &&
	    bpx_unify(bpx_get_call_arg(4,7), bpx_build_float  (em_eng.bic       )) &&
	    bpx_unify(bpx_get_call_arg(5,7), bpx_build_float  (em_eng.cs        )) &&
	    bpx_unify(bpx_get_call_arg(6,7), bpx_build_integer(em_eng.smooth    )) ;
}

/* main loop */
int crf_rank_learn(CRF_ENG_PTR crf_ptr) {
	int r,iterate,old_valid,converged,saved = 0;
	double likelihood,old_likelihood = 0.0;
	double tmp_epsilon,alpha0,gf_sd,old_gf_sd = 0.0;

	config_crf(crf_ptr);

	initialize_weights();

	if (crf_learn_mode == 1) {
		initialize_LBFGS();
		printf("L-BFGS mode\n");
	}

	if (crf_learning_rate==1) {
		printf("learning rate:annealing\n");
	} else if (crf_learning_rate==2) {
		printf("learning rate:backtrack\n");
	} else if (crf_learning_rate==3) {
		printf("learning rate:golden section\n");
	}

	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#crf-iters", r);

		initialize_crf_count();
		initialize_lambdas();
		initialize_visited_flags();

		old_valid = 0;
		iterate = 0;
		tmp_epsilon = crf_epsilon;

		restart_LBFGS();

		while (1) {
			if (CTRLC_PRESSED) {
				SHOW_PROGRESS_INTR();
				RET_ERR(err_ctrl_c_pressed);
			}

			RET_ON_ERR(crf_ptr->compute_feature());

			crf_ptr->compute_crf_probs();

			likelihood = crf_ptr->compute_likelihood();

			if (verb_em) {
				prism_printf("Iteration #%d:\tlog_likelihood=%.9f\n", iterate, likelihood);
			}

			if (debug_level) {
				prism_printf("After I-step[%d]:\n", iterate);
				prism_printf("likelihood = %.9f\n", likelihood);
				print_egraph(debug_level, PRINT_EM);
			}

			if (!std::isfinite(likelihood)) {
				emit_internal_error("invalid log likelihood: %s (at iteration #%d)",
				                    isnan(likelihood) ? "NaN" : "infinity", iterate);
				RET_ERR(ierr_invalid_likelihood);
			}
			/*        if (old_valid && old_likelihood - likelihood > prism_epsilon) {
					  emit_error("log likelihood decreased [old: %.9f, new: %.9f] (at iteration #%d)",
					  old_likelihood, likelihood, iterate);
					  RET_ERR(err_invalid_likelihood);
					  }*/
			if (likelihood > 0.0) {
				emit_error("log likelihood greater than zero [value: %.9f] (at iteration #%d)",
				           likelihood, iterate);
				RET_ERR(err_invalid_likelihood);
			}

			if (crf_learn_mode == 1 && iterate > 0) restore_old_gradient();

			RET_ON_ERR(crf_ptr->compute_gradient());

			if (crf_learn_mode == 1 && iterate > 0) {
				compute_LBFGS_y_rho();
				compute_hessian(iterate);
			} else if (crf_learn_mode == 1 && iterate == 0) {
				initialize_LBFGS_q();
			}

			converged = (old_valid && fabs(likelihood - old_likelihood) <= prism_epsilon);

			if (converged || REACHED_MAX_ITERATE(iterate)) {
				break;
			}

			old_likelihood = likelihood;
			old_valid = 1;

			if (debug_level) {
				prism_printf("After O-step[%d]:\n", iterate);
				print_egraph(debug_level, PRINT_EM);
			}

			SHOW_PROGRESS(iterate);
			
			// computing learning rate
			if (crf_learning_rate == 1) { // annealing
				tmp_epsilon = (annealing_weight / (annealing_weight + iterate)) * crf_epsilon;
			} else if (crf_learning_rate == 2) { // line-search(backtrack)
				// gf_sd = grad f^T dot d (search direction)
				if (crf_learn_mode == 1) {
					gf_sd = compute_gf_sd_LBFGS();
				} else {
					gf_sd = compute_gf_sd();
				}
				if (iterate==0) {
					alpha0 = 1;
				} else {
					alpha0 = tmp_epsilon * old_gf_sd / gf_sd;
				}
				if (crf_learn_mode == 1) {
					tmp_epsilon = line_search_LBFGS(crf_ptr,alpha0,crf_ls_rho,crf_ls_c1,likelihood,gf_sd);
				} else {
					tmp_epsilon = line_search(crf_ptr,alpha0,crf_ls_rho,crf_ls_c1,likelihood,gf_sd);
				}

				if (tmp_epsilon < EPS) {
					emit_error("invalid alpha in line search(=0.0) (at iteration #%d)",iterate);
					RET_ERR(err_line_search);
				}
				old_gf_sd = gf_sd;
			} else if (crf_learning_rate == 3) { // line-search(golden section)
				if (crf_learn_mode == 1) {
					tmp_epsilon = golden_section_LBFGS(crf_ptr,0,crf_golden_b);
				} else {
					tmp_epsilon = golden_section(crf_ptr,0,crf_golden_b);
				}
			}
			// updating with learning rate 
			crf_ptr->update_lambdas(tmp_epsilon);

			iterate++;
		}

		SHOW_PROGRESS_TAIL(converged, iterate, likelihood);

		if (r == 0 || likelihood > crf_ptr->likelihood) {
			crf_ptr->likelihood = likelihood;
			crf_ptr->iterate    = iterate;

			saved = (r < num_restart - 1);
			if (saved) {
				save_params();
			}
		}
	}

	if (crf_learn_mode == 1) clean_LBFGS();
	INIT_VISITED_FLAGS;
	return BP_TRUE;
}


