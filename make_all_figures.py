# -*- coding: utf-8 -*-

"""
This file is ment to reproduce figures for the manuscript
"On the validity of electric brain signal predictions based on population firing rates"
by Torbjørn V. Ness, Tom Tetzlaff, Gaute T. Einevoll, and David Dahmen

"""
import os
from os.path import join
from src import neural_simulations
from src import kernel_validity
from src import brunel_delta_nest

data_folder = join(os.path.dirname(os.path.abspath(__file__)),
                   "simulated_pop_kernels")

toy_results_folder = join(os.path.dirname(os.path.abspath(__file__)),
                          "toy_kernel_results")
simpop_results_folder = join(os.path.dirname(os.path.abspath(__file__)),
                             "simpop_kernel_results")
firing_rate_folder = join(os.path.dirname(os.path.abspath(__file__)),
                          "firing_rate_results")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(toy_results_folder, exist_ok=True)
os.makedirs(simpop_results_folder, exist_ok=True)
os.makedirs(firing_rate_folder, exist_ok=True)

# Figure 3:
kernel_validity.simplest_toy_illustration(toy_results_folder)

# Figure 4:
# kernel_validity.run_illustrative_examples(toy_results_folder)

# Do neural simulations to find kernels
# neural_simulations.kernel_heterogeneities(data_folder)

# Figure 6:
# neural_simulations.make_neural_pop_setup_fig(data_folder)

# Do NEST simulations
# brunel_delta_nest.run_brunel_network(g=5., eta=2.0, J=0.1, order=2500, N_rec=1000, basename="AI_slow")
# brunel_delta_nest.run_brunel_network(g=4.5, eta=0.9, J=0.1, order=2500, N_rec=1000, basename="SI_slow")


# Figure 5
# kernel_validity.run_parameter_scan(toy_results_folder)
# kernel_validity.plot_parameter_scan_results(toy_results_folder)

# kernel_validity.run_parameter_scan_biophysical_model(simpop_results_folder, data_folder, firing_rate_folder)
# Figure 7:
# kernel_validity.investigate_error_measure(data_folder, firing_rate_folder)

# Figure 8:
# kernel_validity.compare_errors("mip_0.0_10_0.0")

# Figure 9:
# kernel_validity.illustrate_firing_rates(firing_rate_folder)

# Figure 10:
# kernel_validity.error_with_correlation_type(data_folder)

# Figure 11:
# kernel_validity.error_summary_figure(data_folder)

# Figure 12:
# kernel_validity.rate_model_figure(data_folder)

# Appendix Figure
# kernel_validity.plot_all_signals_and_kernels(data_folder, firing_rate_folder)

# kernel_validity.gather_pop_kernels(data_folder)

