import os
import sys

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)

import numpy as np
import tensorflow as tf

from DeepSparseCoding.tf1x.data.dataset import Dataset
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap

import response_contour_analysis.utils.model_handling as model_handling
import response_contour_analysis.utils.dataset_generation as iso_data
import response_contour_analysis.utils.histogram_analysis as hist_funcs


def get_dsc_activations_cell(analyzer, images, neuron, batch_size=10, activation_operation=None):
    """
    Returns the activations from a model for given input images
    Parameters:
        analyzer [DSC analyzer object] an object from the DeepSparseCoding library
        images [np.ndarray] of size NumImages x W x H
        neuron [int or vector of ints] that points to the neuron index
        batch_size [int] specifying the batch size to use for the getting the neuron activations
        activation_operation [function] to be used if the DSC model has a unique function handle for getting neuron activations (e.g. in the case of lca_subspace)
    Output:
        activations [np.ndarray] vector of length len(neuron)
    """
    images = dp.reshape_data(images[..., None], flatten=analyzer.model.params.vectorize_data)[0]
    activations = analyzer.compute_activations(images, batch_size, activation_operation)[:, neuron]
    return activations


def load_analyzer(params):
    analyzer = ap.get_analyzer(params.model_type)
    analyzer.setup(params)
    analyzer.model.setup(analyzer.model_params)
    analyzer.load_analysis(save_info=params.save_info)
    return analyzer


class lca_512_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_512_vh"
    self.display_name = "Sparse Coding 512"
    self.version = "0.0"
    #self.save_info = "analysis_train_carlini_targeted"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class lca_768_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_768_vh"
    self.display_name = "Sparse Coding 768"
    self.version = "0.0"
    #self.save_info = "analysis_train_carlini_targeted"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class lca_1024_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_1024_vh"
    self.display_name = "Sparse Coding 1024"
    self.version = "0.0"
    #self.save_info = "analysis_train_carlini_targeted"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class lca_2560_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_2560_vh"
    self.display_name = "Sparse Coding 2560"
    self.version = "0.0"
    #self.save_info = "analysis_train_kurakin_targeted"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class ae_768_vh_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_768_vh"
    self.display_name = "ReLU Autoencoder 768"
    self.version = "1.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class sae_768_vh_params(object):
  def __init__(self):
    self.model_type = "sae"
    self.model_name = "sae_768_vh"
    self.display_name = "Sparse Autoencoder 768"
    self.version = "1.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class rica_768_vh_params(object):
  def __init__(self):
    self.model_type = "rica"
    self.model_name = "rica_768_vh"
    self.display_name = "Linear Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class lca_768_mnist_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_768_mnist"
    self.display_name = "Sparse Coding 768"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class lca_1536_mnist_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_1536_mnist"
    self.display_name = "Sparse Coding 1536"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class ae_768_mnist_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_768_mnist"
    self.display_name = "ReLU Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class sae_768_mnist_params(object):
  def __init__(self):
    self.model_type = "sae"
    self.model_name = "sae_768_mnist"
    self.display_name = "Sparse Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class rica_768_mnist_params(object):
  def __init__(self):
    self.model_type = "rica"
    self.model_name = "rica_768_mnist"
    self.display_name = "Linear Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class ae_deep_mnist_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_deep_mnist"
    self.display_name = "ReLU Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = False
    self.model_dir = (root_path+'/Projects/'+self.model_name)


class lca_subspace_params(object):
  def __init__(self):
    self.model_type = "lca_subspace"
    self.model_name = "lca_subspace_vh"
    self.display_name = "SSC"
    self.version = "3.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False
    self.use_group_activations = True
    self.model_dir = (root_path+'/Projects/'+self.model_name)


if __name__ == "__main__":
    print("Loading models...")
    cont_analysis = {}
    cont_analysis['min_angle'] = 15
    cont_analysis['batch_size'] = 100
    cont_analysis['vh_image_scale'] = 31.773287 # Mean of the l2 norm of the training set
    cont_analysis['comparison_method'] = 'closest' # rand or closest

    cont_analysis['num_neurons'] = 100 # How many neurons to plot
    cont_analysis['num_comparisons'] = 300 # How many planes to construct (None is all of them)
    cont_analysis['x_range'] = [-2.0, 2.0]
    cont_analysis['y_range'] = [-2.0, 2.0]
    cont_analysis['num_images'] = int(30**2)

    cont_analysis['params_list'] = [lca_512_vh_params()]
    #cont_analysis['params_list'] = [lca_768_vh_params()]
    #cont_analysis['params_list'] = [lca_1024_vh_params()]
    #cont_analysis['params_list'] = [lca_2560_vh_params()]

    #cont_analysis['iso_save_name'] = "iso_curvature_xrange1.3_yrange-2.2_"
    #cont_analysis['iso_save_name'] = "iso_curvature_ryan_"
    cont_analysis['iso_save_name'] = "rescaled_closecomp_"
    #cont_analysis['iso_save_name'] = ''

    np.savez(save_root+'iso_params_'+cont_analysis['iso_save_name']+params.save_info+".npz",
        data=cont_analysis)

    analyzer_list = [load_analyzer(params) for params in cont_analysis['params_list']]
    for analyzer, params in zip(analyzer_list, cont_analysis['params_list']):
      print(analyzer.analysis_params.display_name)
      print("Computing the iso-response vectors...")
      cont_analysis['target_neuron_ids'] = iso_data.get_rand_target_neuron_ids(
          cont_analysis['num_neurons'], analyzer.model.params.num_neurons)
      neuron_weights = [analyzer.bf_stats["basis_functions"][idx]
          for idx in range(len(analyzer.bf_stats["basis_functions"]))]
      analyzer.target_neuron_ids = cont_analysis['target_neuron_ids']
      rand_outputs = iso_data.compute_rand_vectors(
          neuron_weights,
          cont_analysis["num_comparisons"])
      analyzer.rand_target_vectors = rand_outputs[0]
      analyzer.rand_orth_vectors = rand_outputs[1]
      comp_outputs = iso_data.compute_comp_vectors(
          neuron_weights,
          cont_analysis['target_neuron_ids'],
          cont_analysis['min_angle'],
          cont_analysis['num_comparisons'],
          cont_analysis['comparison_method'])
      analyzer.comparison_neuron_ids = comp_outputs[0]
      analyzer.comparison_target_vectors = comp_outputs[1]
      analyzer.comparison_vectors = comp_outputs[2]
      analyzer.target_vectors = analyzer.comparison_target_vectors
      assert len(analyzer.comparison_neuron_ids) == cont_analysis['num_neurons'], (
          "Incorrect number of comparison vectors")
      for comparison_ids_list in analyzer.comparison_neuron_ids:
          assert len(comparison_ids_list) >= cont_analysis['num_comparisons'], (
              "Not enough comparison vectors.")
      key_list = ["target_neuron_ids", "comparison_neuron_ids", "target_vectors",
          "rand_orth_vectors", "comparison_vectors"]
      val_list = [analyzer.target_neuron_ids, analyzer.comparison_neuron_ids, analyzer.target_vectors,
          analyzer.rand_orth_vectors, analyzer.comparison_vectors]
      iso_vectors = dict(zip(key_list, val_list))
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_vectors_"+cont_analysis['iso_save_name']+params.save_info+".npz",
          data=iso_vectors)
      for use_rand_orth_vects, rand_str in zip([True, False], ["rand", "comparison"]):
          print("Generating "+rand_str+" dataset...")
          comp_vects = analyzer.rand_orth_vectors if use_rand_orth_vects else analyzer.comparison_vectors
          contour_dataset, datapoints = iso_data.get_contour_dataset(
              analyzer.target_vectors, comp_vects, cont_analysis['x_range'], cont_analysis['y_range'],
              cont_analysis['num_images'], cont_analysis['vh_image_scale'])
          print("Computing network activations for "+rand_str+" dataset...")
          if params.use_group_activations:
              activation_operation = analyzer.model.get_reshaped_group_activity
          else:
              activation_operation = None
          activation_function_kwargs = {
              'activation_operation': activation_operation,
              'batch_size': cont_analysis['batch_size']
          }
          activations = model_handling.get_normalized_activations(
              analyzer,
              cont_analysis["target_neuron_ids"],
              datapoints,
              get_dsc_activations_cell,
              activation_function_kwargs)
          save_root=analyzer.analysis_out_dir+'savefiles/'
          if use_rand_orth_vects:
              np.savez(save_root+'iso_rand_activations_'+cont_analysis['iso_save_name']+params.save_info+'.npz',
                  data=activations)
              np.savez(save_root+'iso_rand_contour_dataset_'+cont_analysis['iso_save_name']+params.save_info+'.npz',
                  data=contour_dataset)
          else:
              np.savez(save_root+'iso_comp_activations_'+cont_analysis['iso_save_name']+params.save_info+'.npz',
                  data=activations)
              np.savez(save_root+'iso_comp_contour_dataset_'+cont_analysis['iso_save_name']+params.save_info+'.npz',
                  data=contour_dataset)
              cont_analysis['comparison_neuron_ids'] = analyzer.comparison_neuron_ids
              cont_analysis['contour_dataset'] = contour_dataset
              curvatures, fits = hist_funcs.iso_response_curvature_poly_fits(
                cont_analysis['activations'],
                target_act=cont_analysis['target_act'],
                measure_upper_right=False
              )
              cont_analysis['curvatures'] = np.stack(np.stack(curvatures, axis=0), axis=0)
              np.savez(save_root+'group_iso_vectors_'+cont_analysis['iso_save_name']+params.save_info+'.npz',
                  data=cont_analysis)
