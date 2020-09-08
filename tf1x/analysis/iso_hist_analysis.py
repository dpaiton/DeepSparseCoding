import os
import sys

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)

import numpy as np

import DeepSparseCoding.tf1x.analysis.analysis_picker as ap
import response_contour_analysis.utils.histogram_analysis as ha
import response_contour_analysis.utils.dataset_generation as dg


class lca_512_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_512_vh"
    self.display_name = "Sparse Coding 512"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False


class lca_768_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_768_vh"
    self.display_name = "Sparse Coding"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False


class lca_1024_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_1024_vh"
    self.display_name = "Sparse Coding 1024"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False


class lca_2560_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_2560_vh"
    self.display_name = "Sparse Coding"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False


def load_analysis(params):
    params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+params.model_name)
    analyzer = ap.get_analyzer(params.model_type)
    analyzer.setup(params)
    analyzer.model.setup(analyzer.model_params)
    analyzer.load_analysis(save_info=params.save_info)
    analyzer.model_name = params.model_name
    return analyzer


params_list = [lca_512_vh_params(), lca_1024_vh_params(), lca_2560_vh_params()]
iso_save_name = 'rescaled_randomcomp_'#"iso_curvature_xrange1.3_yrange-2.2_"
#attn_save_name = 'rescaled_randomcomp_'#'1d_'

num_bins = 50
target_act = 0.5

for params in params_list:
  params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+params.model_name)

analyzer_list = [ap.get_analyzer(params.model_type) for params in params_list]
for analyzer, params in zip(analyzer_list, params_list):
    analyzer.setup(params)
    analyzer.model.setup(analyzer.model_params)
    analyzer.load_analysis(save_info=params.save_info)
    analyzer.model_name = params.model_name
    analyzer.iso_params = np.load(
        analyzer.analysis_out_dir+"savefiles/iso_params_"
        +iso_save_name+analyzer.analysis_params.save_info+".npz",
        allow_pickle=True)["data"].item()
    analyzer.iso_comp_activations = np.load(
        analyzer.analysis_out_dir+"savefiles/iso_comp_activations_"
        +iso_save_name+analyzer.analysis_params.save_info+".npz",
        allow_pickle=True)["data"]
    analyzer.iso_comp_contour_dataset = np.load(
        analyzer.analysis_out_dir+"savefiles/iso_comp_contour_dataset_"
        +iso_save_name+analyzer.analysis_params.save_info+".npz",
        allow_pickle=True)["data"].item()
    analyzer.iso_rand_activations = np.load(
        analyzer.analysis_out_dir+"savefiles/iso_rand_activations_"
        +iso_save_name+analyzer.analysis_params.save_info+".npz",
        allow_pickle=True)["data"]
    analyzer.iso_rand_contour_dataset = np.load(
        analyzer.analysis_out_dir+"savefiles/iso_rand_contour_dataset_"
        +iso_save_name+analyzer.analysis_params.save_info+".npz",
        allow_pickle=True)["data"].item()
    analyzer.iso_num_target_neurons = analyzer.iso_params["num_neurons"]
    analyzer.iso_num_comparison_vectors = analyzer.iso_params["num_comparisons"]
    output_dict = {}
    iso_comp_out = ha.iso_response_curvature_poly_fits(
        analyzer.iso_comp_activations,
        target_act,
        measure_upper_right=True
    )
    output_dict['iso_comp_curvatures'], output_dict['iso_comp_fits'] = iso_comp_out
    analyzer.iso_comp_curvatures, analyzer.iso_comp_fits = iso_comp_out
    iso_rand_out = ha.iso_response_curvature_poly_fits(
        analyzer.iso_rand_activations,
        target_act,
        measure_upper_right=True
    )
    output_dict['iso_rand_curvatures'], output_dict['iso_rand_fits'] = iso_rand_out
    analyzer.iso_rand_curvatures, analyzer.iso_rand_fits = iso_rand_out
    attn_comp_out = ha.response_attenuation_curvature_poly_fits(
        analyzer.iso_comp_activations,
        target_act,
        analyzer.iso_comp_contour_dataset['x_pts'],
        analyzer.iso_comp_contour_dataset['proj_datapoints']
    )
    output_dict['attn_comp_curvatures'], output_dict['attn_comp_fits'], output_dict['attn_comp_sliced_activity']  = attn_comp_out
    analyzer.attn_comp_curvatures, analyzer.attn_comp_fits, analyzer.attn_comp_sliced_activity = attn_comp_out
    attn_rand_out = ha.response_attenuation_curvature_poly_fits(
        analyzer.iso_rand_activations,
        target_act,
        analyzer.iso_rand_contour_dataset['x_pts'],
        analyzer.iso_rand_contour_dataset['proj_datapoints']
    )
    output_dict['attn_rand_curvatures'], output_dict['attn_rand_fits'], output_dict['attn_rand_sliced_activity'] = attn_rand_out
    analyzer.attn_rand_curvatures, analyzer.attn_rand_fits, analyzer.attn_rand_sliced_activity = attn_rand_out
    #curvatures [nested list of floats] that is indexed by
    #    [curvature type]
    #    [dataset type]
    #    [target neuron id]
    #    [comparison plane id]
    curvatures = [
        [ # iso
            [ # comp
                output_dict['iso_comp_curvatures']
            ], [ # rand
                output_dict['iso_rand_curvatures']
            ]
        ], [ # attn
            [ # comp
                output_dict['attn_comp_curvatures']
            ], [ # rand
                output_dict['attn_rand_curvatures']
            ]
        ]
    ]
    curvature_hists_out = ha.compute_curvature_hists(curvatures, num_bins)
    output_dict['all_hists'], output_dict['all_bin_edges'] = curvature_hists_out
    analyzer.all_hists, analyzer.all_bin_edges = curvature_hists_out
    all_bin_centers = []
    for bin_edges in analyzer.all_bin_edges:
        bin_lefts, bin_rights = bin_edges[:-1], bin_edges[1:]
        bin_centers = bin_lefts + (bin_rights - bin_lefts)
        all_bin_centers.append(bin_centers)
    output_dict['all_bin_centers'] = all_bin_centers
    analyzer.all_bin_centers = all_bin_centers
    np.savez(analyzer.analysis_out_dir+"savefiles/iso_hists_"+iso_save_name+params.save_info+".npz",
      data=output_dict)
