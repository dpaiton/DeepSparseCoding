"""
Settings used for generating NIPS plots
"""
import sys
topleveldir = '/home/spencerkent/Projects/Sophias_DeepSparseCoding'
sys.path.append(topleveldir)

import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from gabor_fit import fit as g_fit
import analysis.analysis_picker as ap
import utils.data_processing as dp
import utils.plot_functions as pf
import spencers_stuff.metrics as sm
import spencers_stuff.plotting as sp
import spencers_stuff.grating_fit as grating_fit

basis_func_dimensions = {'height': 16, 'width': 16}
num_phase_samples = 32
num_orientation_samples = 32

# Positive-only LCA PARAMS
# analyzer_params = {
#     'model_type': 'lca',
#     'model_name': 'lca_vh_ft_1c',
#     'version': '0.0',
#     'device': '/gpu:0',
#     'save_info': 'analysis_analysis',
#     'ft_padding': 16,
#     'neuron_indices': None,
#     'checkpoint_filename': 'lca_vh_ft_1c_v0.0_full-100000',
#     'contrasts': [0.2, 0.4, 0.6, 0.8, 1.0],
#     'phases': np.arange(-16, 16, 32/num_phase_samples),
#     'orientations': np.arange(-np.pi/2, np.pi/2, np.pi/num_orientation_samples)}
#
# analyzer_params['model_dir'] = \
#     '/media/expansion1/spencerkent/logfiles/DeepSparseCoding_visualization/model_dir/lca_vh_ft_1c'
# dummy_images = np.zeros((100, 256))
# print("Loading analyzer object")
# analyzer = ap.get_analyzer(analyzer_params)
# analyzer.model.setup(analyzer.model_params, analyzer.model_schedule)
# learned_basis_funcs = analyzer.evaluate_model(dummy_images, ["weights/phi:0"])["weights/phi:0"]


# LCA PARAMS
# analyzer_params = {
#     'model_type': 'lca',
#     'model_name': 'lca_256_l0_2.5',
#     'version': '1.0',
#     'device': '/gpu:0',
#     'save_info': 'analysis_analysis',
#     'ft_padding': 16,
#     'neuron_indices': None,
#     'checkpoint_filename': 'lca_256_l0_2.5_v1.0_full-1000000',
#     'contrasts': [0.2, 0.4, 0.6, 0.8, 1.0],
#     'phases': np.arange(-16, 16, 32/num_phase_samples),
#     'orientations': np.arange(-np.pi/2, np.pi/2, np.pi/num_orientation_samples)}
#
# analyzer_params['model_dir'] = \
#     '/media/expansion1/spencerkent/logfiles/DeepSparseCoding_visualization/model_dir/lca_256_l0_2.5'
# dummy_images = np.zeros((100, 256))
# print("Loading analyzer object")
# analyzer = ap.get_analyzer(analyzer_params)
# analyzer.model.setup(analyzer.model_params, analyzer.model_schedule)
# learned_basis_funcs = analyzer.evaluate_model(dummy_images, ["weights/phi:0"])["weights/phi:0"]


# ICA PARAMS
analyzer_params = {
    'model_type': 'ica',
    'model_name': 'ica_v1.0',
    'version': '1.0',
    'device': '/gpu:0',
    'save_info': 'analysis_analysis',
    'ft_padding': 16,
    'neuron_indices': None,
    'checkpoint_filename': 'ica_v1.0_v1.0_full-1000000',
    'contrasts': [0.2, 0.4, 0.6, 0.8, 1.0],
    'phases': np.arange(-16, 16, 32/num_phase_samples),
    'orientations': np.arange(-np.pi/2, np.pi/2, np.pi/num_orientation_samples)}

analyzer_params['model_dir'] = \
    '/media/expansion1/spencerkent/logfiles/DeepSparseCoding_visualization/model_dir/ica_v1.0'
dummy_images = np.zeros((100, 256))
print("Loading analyzer object")
analyzer = ap.get_analyzer(analyzer_params)
analyzer.model.setup(analyzer.model_params, analyzer.model_schedule)
learned_basis_funcs = analyzer.evaluate_model(dummy_images, ["weights/a_inverse:0"])["weights/a_inverse:0"]
input('wait here')


# okay now this should be common to all models

# reshaped_bf = np.reshape(learned_basis_funcs.T, (learned_basis_funcs.shape[1], 16, 16))
# gabor_fit_params = pickle.load(open(analyzer_params['model_dir'] + 
#                                     '/savefiles/params_of_well_fit_neurons.p',
#                                     'rb'))
# test_neuron_inds = pickle.load(open(analyzer_params['model_dir'] + 
#                                     '/savefiles/indices_of_well_fit_neurons.p',
#                                     'rb'))
# print("len test_neuron_inds: ", len(test_neuron_inds))
# print("len gabor_fit_params: ", len(gabor_fit_params))
# print("num test neurons: ", len(test_neuron_inds))
# # let's generate the gabors and take a look at them
# fitter = g_fit.GaborFit()
# tneuron_gabor_synth = np.zeros((len(test_neuron_inds), 16, 16))
# tneuron_envelope_synth = np.zeros((len(test_neuron_inds), 16, 16))
# tneuron_grating_synth = np.zeros((len(test_neuron_inds), 16, 16))
# envelope_integral = []
# for bf_idx in range(len(test_neuron_inds)):
#   tneuron_gabor_synth[bf_idx] = np.squeeze(
#       fitter.make_gabor(gabor_fit_params[bf_idx], 16, 16))
#   tneuron_envelope_synth[bf_idx] = np.squeeze(
#       fitter.make_envelope(gabor_fit_params[bf_idx], 16, 16))
#   tneuron_grating_synth[bf_idx] = np.squeeze(
#       fitter.make_phase(gabor_fit_params[bf_idx], 16, 16))
#   envelope_integral.append(np.sum(tneuron_envelope_synth[bf_idx]))
#
# # Let's just get rid of filters who's envelopes are wacky
# problematic_envelope_fits = np.where(
#     np.logical_or(np.array(envelope_integral) < 5.0,
#                   np.array(envelope_integral) > 11.0))[0]
# disqualified_during_fitting = (set([x for x in range(reshaped_bf.shape[0])]) -
#                                set(test_neuron_inds))
#
# # all_bfunc_fig = sp.plot_basis_functions(
# #     reshaped_bf, plot_title="Original basis functions w/ disqualifications",
# #     freq_dq_inds=disqualified_during_fitting, env_dq_inds=np.array(test_neuron_inds)[problematic_envelope_fits],
# #     renormalize=True, single_img=False)
#
#
# # okay if these all look good we can compute the curves
# final_test_n_inds = [test_neuron_inds[x] for x in range(len(test_neuron_inds))
#                      if x not in problematic_envelope_fits]
# final_gabor_params = [gabor_fit_params[x] for x in range(len(gabor_fit_params))
#                       if x not in problematic_envelope_fits]
# # bfunc_fig = sp.plot_basis_functions(
# #     tneuron_gabor_synth[[x for x in range(len(test_neuron_inds)) if x not in
# #       problematic_envelope_fits]], plot_title="synth basis functions w/ disqualifications",
# #     renormalize=True, single_img=True)
# #
# # plt.show()
# # print("Final test indexes: ", final_test_n_inds)
# # print("Final test indexes: ", final_test_n_inds)
#
# orientation_tuning_responses = analyzer.spencers_orientation_tuning(
#     16, final_test_n_inds, final_gabor_params, global_contrast=6.0, 
#     fit_type='gabor')
#
# desired_metrics = {
#     'abs_mean_over_phase': ['full width half maximum', 'circular variance',
#                             'orientation selectivity index'],
#     'abs_max_over_phase': ['full width half maximum', 'circular variance',
#                            'orientation selectivity index']}
#     # 'pos_mean_over_phase': ['full width half maximum', 'circular variance',
#     #                         'orientation selectivity index'],
#     # 'pos_max_over_phase': ['full width half maximum', 'circular variance',
#     #                        'orientation selectivity index']}
#
# orientation_tuning_metrics = sm.compute_ot_metrics(
#     orientation_tuning_responses, desired_metrics,
#     corresponding_angles_deg=(180 * np.arange(num_orientation_samples) /
#                               num_orientation_samples) - 90,
#     corresponding_angles_rad=(np.pi * np.arange(num_orientation_samples) /
#                               num_orientation_samples) - (np.pi/2))
#
# curve_figs = []
# cv_figs = []
# for ot_variant in orientation_tuning_responses:
#   curve_figs.append(sp.plot_ot_curves(
#     orientation_tuning_responses[ot_variant],
#     {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0},
#     metrics=orientation_tuning_metrics[ot_variant],
#     plot_title="Orientation tuning curves, variant " + ot_variant))
#   cv_figs.append(sp.plot_circular_variance(
#     orientation_tuning_metrics[ot_variant]['circular variance'],
#     plot_title='Circular variance visualization, variant ' + ot_variant))
#   aggregate_fwhm = [x[1] - x[0] for x in 
#       orientation_tuning_metrics[ot_variant]['full width half maximum']]
#   aggregate_cv = [x[2] for x in 
#       orientation_tuning_metrics[ot_variant]['circular variance']]
#   # print("all cvs for variant " + ot_variant)
#   # print(aggregate_cv)
#   # pickle.dump({'fwhm': aggregate_fwhm, 'cv': aggregate_cv},
#   #             open(analyzer_params['model_dir'] + 
#   #                  '/savefiles/aggregate_metrics_gabor_' + ot_variant +
#   #                  '.p', 'wb'))
#   # pickle.dump(orientation_tuning_responses, open(analyzer_params['model_dir'] +
#   #   '/savefiles/full_set_of_tuning_curves_gabor_' + ot_variant + '.p', 'wb'))
#
#
#
# dq_bfunc_fig = sp.plot_basis_functions(
#     reshaped_bf[final_test_n_inds],
#     plot_title="Corresponding basis functions",
#     renormalize=True, reference_interval = [0., 1.], single_img=True)
#
# plt.show()
