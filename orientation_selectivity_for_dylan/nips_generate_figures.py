"""
Code use to generate orientation selectivity figures
"""
import sys
topleveldir = '/home/spencerkent/Projects/Sophias_DeepSparseCoding'
sys.path.append(topleveldir)

import pickle
import numpy as np
from matplotlib import pyplot as plt
import spencers_stuff.metrics as sm
import spencers_stuff.plotting as sp


# let's first show a random size-25 sample of the tuning curves at full 
# contrast for each of the two models

lca_dir = \
    '/media/expansion1/spencerkent/logfiles/DeepSparseCoding_visualization/model_dir/lca_256_l0_2.5'
lca_curves = pickle.load(open(lca_dir + 
  '/savefiles/full_set_of_tuning_curves_gabor_abs_mean_over_phase.p', 'rb'))['abs_mean_over_phase']
#^ bug in previous saved both variants

random_lca_samps = np.random.choice(np.arange(lca_curves.shape[0]), 25, replace=False)
sampled_lca_metrics = sm.compute_ot_metrics({'lca_curves':lca_curves[random_lca_samps]},
    which_metrics={'lca_curves': ['full width half maximum', 'circular variance']},
    corresponding_angles_deg=(180 * np.arange(32) / 32) - 90,
    corresponding_angles_rad=(np.pi * np.arange(32) / 32) - (np.pi/2))

tuning_fig_lca = sp.plot_ot_curves(
    lca_curves[random_lca_samps],
    {4: 1.0},
    metrics=sampled_lca_metrics['lca_curves'],
    plot_title="Random sampling of LCA tuning curves")

ica_dir = \
    '/media/expansion1/spencerkent/logfiles/DeepSparseCoding_visualization/model_dir/ica_v1.0'
ica_curves = pickle.load(open(ica_dir + 
  '/savefiles/full_set_of_tuning_curves_gabor_abs_mean_over_phase.p', 'rb'))['abs_mean_over_phase']
#^ bug in previous saved both variants

random_ica_samps = np.random.choice(np.arange(ica_curves.shape[0]), 25, replace=False)
sampled_ica_metrics = sm.compute_ot_metrics({'ica_curves':ica_curves[random_ica_samps]},
    which_metrics={'ica_curves': ['full width half maximum', 'circular variance']},
    corresponding_angles_deg=(180 * np.arange(32) / 32) - 90,
    corresponding_angles_rad=(np.pi * np.arange(32) / 32) - (np.pi/2))

tuning_fig_ica = sp.plot_ot_curves(
    ica_curves[random_ica_samps],
    {4: 1.0},
    metrics=sampled_ica_metrics['ica_curves'],
    plot_title="Random sampling of ICA tuning curves")

plt.show()

# aggregate_metrics_LCA = pickle.load(open(lca_dir +
#     '/savefiles/aggregate_metrics_gabor_abs_mean_over_phase.p', 'rb'))
# aggregate_metrics_ICA = pickle.load(open(ica_dir +
#     '/savefiles/aggregate_metrics_gabor_abs_mean_over_phase.p', 'rb'))
# fwhm_density = sp.compare_empirical_density(
#     [aggregate_metrics_LCA['fwhm'], aggregate_metrics_ICA['fwhm']],
#     ['LCA', 'ICA'], 30, lines=False)
# cv_density = sp.compare_empirical_density(
#     [aggregate_metrics_LCA['cv'], aggregate_metrics_ICA['cv']],
#     ['LCA', 'ICA'], 30, lines=False)
# plt.show()


