import os
import re
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
import data.data_selector as ds
import utils.data_processing as dp
import utils.plot_functions as pf
import analysis.analysis_picker as ap

def compute_iso_vectors(analyzer, min_angle, max_angle, num_neurons, use_bf_stats):
  """
  Calculate all projection vectors for each target neuron
  For each target neuron, build dataset of random orthogonal vectors & selected orthogonal vectors
  """
  # Get list of candidate target vectors
  if(use_bf_stats):
    neuron_angles, plot_matrix = analyzer.get_neuron_angles(analyzer.bf_stats)
  else:
    optimal_stims_dict = {"patch_edge_size":int(np.sqrt(analyzer.model.params.data_shape)),
      "num_outputs":len(analyzer.analysis_params.neuron_vis_targets),
      "basis_functions":[]}
    for target_id in range(len(analyzer.analysis_params.neuron_vis_targets)):
      bf = analyzer.neuron_vis_output["optimal_stims"][target_id][-1]
      optimal_stims_dict["basis_functions"].append(bf.reshape((optimal_stims_dict["patch_edge_size"],)*2))
    neuron_angles, plot_matrix = analyzer.get_neuron_angles(optimal_stims_dict)
  orig_min_angle = min_angle
  orig_max_angle = max_angle
  vectors = np.argwhere(np.logical_and(plot_matrix<max_angle, plot_matrix>min_angle))
  num_tries=0
  while len(set(vectors[:,0])) <= num_neurons:
    vectors = np.argwhere(np.logical_and(plot_matrix<max_angle, plot_matrix>min_angle))
    if min_angle > 5:
      min_angle -= 1
    if max_angle < 89:
      max_angle += 1
    num_tries += 1
    if num_tries > 100:
      print("Unable to find comparison vectors...")
      import IPython; IPython.embed(); raise SystemExit
  if min_angle < orig_min_angle or max_angle > orig_max_angle:
    print("compute_iso_vectors:WARNING:"
      +"The provided angle range was too small, the new angle range is [%g, %g]"%(min_angle,
      max_angle))
  target_neuron_ids = []
  comparison_neuron_ids = [] # list of lists [num_targets][num_comparisons_per_target]
  target_vectors = []
  rand_orth_vectors = []
  comparison_vectors = []
  unique_vectors = list(set(vectors[:,0]))
  for vector_set_id in range(num_neurons):
    vector_id = np.argwhere(vectors[:,0] == unique_vectors[vector_set_id])[0]
    target_neuron_id = vectors[vector_id, 0].item()
    target_neuron_ids.append(target_neuron_id)
    # Reshape & rescale target vector
    target_vector = analyzer.bf_stats["basis_functions"][target_neuron_id]
    target_vector = target_vector.reshape(analyzer.model_params.num_pixels)
    target_vector = target_vector / np.linalg.norm(target_vector)
    target_vectors.append(target_vector)
    # Build matrix of random orthogonal vectors
    rand_orth_vectors.append(dp.get_rand_orth_vectors(target_vector,
      analyzer.model.params.num_pixels-1))
    # Build matrix of comparison vectors (use all neurons)
    if(use_bf_stats):
      sub_comparison_neuron_ids = [vectors[vector_id, 1].item()]
      for index in range(analyzer.bf_stats["num_outputs"]):
        if index != target_neuron_id and index not in sub_comparison_neuron_ids:
          sub_comparison_neuron_ids.append(index)
    else:
      sub_comparison_neuron_ids = [index for index in range(optimal_stims_dict["num_outputs"])
        if index != target_neuron_id]
    comparison_vector_matrix = target_vector.T[:,None] # matrix of alternate vectors
    for comparison_neuron_id in sub_comparison_neuron_ids:
      if(use_bf_stats):
        comparison_vector = analyzer.bf_stats["basis_functions"][comparison_neuron_id]
      else:
        comparison_vector = optimal_stims_dict["basis_functions"][comparison_neuron_id]
      comparison_vector = comparison_vector.reshape(analyzer.model_params.num_pixels)
      comparison_vector = np.squeeze((comparison_vector / np.linalg.norm(comparison_vector)).T)
      comparison_vector_matrix = np.append(comparison_vector_matrix, comparison_vector[:,None],
        axis=1)
    comparison_neuron_ids.append(sub_comparison_neuron_ids)
    comparison_vectors.append(comparison_vector_matrix.T[1:,:])
  return (target_neuron_ids, comparison_neuron_ids, target_vectors, rand_orth_vectors, comparison_vectors)

def get_contour_dataset(analyzer, num_comparison_vects, use_random_orth_vects, x_range, y_range, num_images):
  """
  datapoints has shape [num_target_neurons][num_comparisons_per_target (or num_planes)][num_datapoints, datapoint_length]
  """
  x_pts = np.linspace(x_range[0], x_range[1], int(np.sqrt(num_images)))
  y_pts = np.linspace(y_range[0], y_range[1], int(np.sqrt(num_images)))
  X_mesh, Y_mesh = np.meshgrid(x_pts, y_pts)
  proj_datapoints = np.stack([X_mesh.reshape(num_images), Y_mesh.reshape(num_images)], axis=1)
  all_datapoints = []
  out_dict = {
    "proj_target_neuron": [],
    "proj_comparison_neuron": [],
    "proj_orth_vect": [],
    "orth_vect": [],
    "proj_datapoints": proj_datapoints,
    "x_pts": x_pts,
    "y_pts": y_pts}
  if use_random_orth_vects:
    comparison_vectors = analyzer.rand_orth_vectors
  else:
    comparison_vectors = analyzer.comparison_vectors
  for target_vect, all_comparison_vects in zip(analyzer.target_vectors, comparison_vectors):
    proj_target_neuron_sub_list = []
    proj_comparison_neuron_sub_list = []
    proj_orth_vect_sub_list = []
    orth_vect_sub_list = []
    datapoints_sub_list = []
    if num_comparison_vects is None or num_comparison_vects > all_comparison_vects.shape[0]-1:
        num_comparison_vects = all_comparison_vects.shape[0]
    for comparison_vect_idx in range(num_comparison_vects): # Each contour plane for the population study
      comparison_vect = np.squeeze(all_comparison_vects[comparison_vect_idx, :])
      proj_matrix, orth_vect = dp.bf_projections(target_vect, comparison_vect)
      proj_target_neuron_sub_list.append(np.dot(proj_matrix, target_vect).T) #project
      proj_comparison_neuron_sub_list.append(np.dot(proj_matrix, comparison_vect).T) #project
      proj_orth_vect_sub_list.append(np.dot(proj_matrix, orth_vect).T) #project
      orth_vect_sub_list.append(orth_vect)
      datapoints = np.stack([np.dot(proj_matrix.T, proj_datapoints[data_id,:])
        for data_id in range(num_images)], axis=0) #inject
      datapoints = dp.reshape_data(datapoints, flatten=False)[0]
      datapoints = {"test": Dataset(datapoints, lbls=None, ignore_lbls=None, rand_state=analyzer.rand_state)}
      datapoints = analyzer.model.reshape_dataset(datapoints, analyzer.model_params)
      datapoints_sub_list.append(datapoints)
    all_datapoints.append(datapoints_sub_list)
    out_dict["proj_target_neuron"].append(proj_target_neuron_sub_list)
    out_dict["proj_comparison_neuron"].append(proj_comparison_neuron_sub_list)
    out_dict["proj_orth_vect"].append(proj_orth_vect_sub_list)
    out_dict["orth_vect"].append(orth_vect_sub_list)
  return out_dict, all_datapoints

def get_normalized_activations(analyzer, contour_dataset):
  """
  contour_dataset should have shape [num_target_neurons][num_comparisons_per_target][num_datapoints, datapoint_length]
  output list is shape [num_target_neurons][num_comparisons_per_target][num_datapoints_x, num_datapoints_y]

  # TODO: Verify batch size is working for compute_activations
  """
  activations_list = []
  for target_index, neuron_index in enumerate(analyzer.target_neuron_ids):
    activity_sub_list = []
    for comparison_index, datapoints in enumerate(contour_dataset[target_index]):
      num_images, data_size = datapoints["test"].images.shape
      batch_size = 1
      for n in range(2, num_images):
        if num_images%n == 0:
          batch_size = num_images // n # second greatest factor
          break
      activations = analyzer.compute_activations(datapoints["test"].images, batch_size)
      activations = activations[:, neuron_index]
      activity_max = np.amax(np.abs(activations))
      activations = activations / (activity_max + 0.00001)
      activations = activations.reshape(int(np.sqrt(num_images)), int(np.sqrt(num_images)))
      activity_sub_list.append(activations)
    activations_list.append(np.stack(activity_sub_list, axis=0))
  return np.stack(activations_list, axis=0)

class lca_512_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_512_vh"
    self.display_name = "Sparse Coding"
    self.version = "0.0"
    self.save_info = "analysis_train_carlini_targeted"
    self.overwrite_analysis_log = False

class lca_768_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_768_vh"
    self.display_name = "Sparse Coding"
    self.version = "0.0"
    self.save_info = "analysis_train_carlini_targeted"
    self.overwrite_analysis_log = False

class lca_1024_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_1024_vh"
    self.display_name = "Sparse Coding"
    self.version = "0.0"
    self.save_info = "analysis_train_carlini_targeted"
    self.overwrite_analysis_log = False

class ae_768_vh_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_768_vh"
    self.display_name = "ReLU Autoencoder"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class sae_768_vh_params(object):
  def __init__(self):
    self.model_type = "sae"
    self.model_name = "sae_768_vh"
    self.display_name = "Sparse Autoencoder"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class rica_768_vh_params(object):
  def __init__(self):
    self.model_type = "rica"
    self.model_name = "rica_768_vh"
    self.display_name = "Linear Autoencoder"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class lca_768_mnist_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_768_mnist"
    self.display_name = "Sparse Coding"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class lca_1536_mnist_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_1536_mnist"
    self.display_name = "Sparse Coding 1536"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False

class ae_768_mnist_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_768_mnist"
    self.display_name = "ReLU Autoencoder"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False

class sae_768_mnist_params(object):
  def __init__(self):
    self.model_type = "sae"
    self.model_name = "sae_768_mnist"
    self.display_name = "Sparse Autoencoder"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False

class rica_768_mnist_params(object):
  def __init__(self):
    self.model_type = "rica"
    self.model_name = "rica_768_mnist"
    self.display_name = "Linear Autoencoder"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class ae_deep_mnist_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_deep_mnist"
    self.display_name = "ReLU Autoencoder"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False

print("Loading models...")
min_angle = 15
max_angle = 65
num_neurons = 2 # How many neurons to plot
use_bf_stats = True # If false, then use optimal stimulus
num_comparison_vects = None # How many planes to construct (None is all of them)
x_range = [-2, 2]
y_range = [-2, 2]
num_images = int(20**2)

params_list = [lca_768_mnist_params(), lca_1536_mnist_params()]
#params_list = [lca_512_vh_params(), lca_768_vh_params(), lca_1024_vh_params()]
#params_list = [rica_768_vh_params(), ae_768_vh_params(), sae_768_vh_params()]#, lca_768_vh_params()]
#params_list = [rica_768_mnist_params(), ae_768_mnist_params(), sae_768_mnist_params(), lca_768_mnist_params()]
for params in params_list:
  params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+params.model_name)
analyzer_list = [ap.get_analyzer(params.model_type) for params in params_list]
for analyzer, params in zip(analyzer_list, params_list):
  analyzer.setup(params)
  analyzer.model.setup(analyzer.model_params)
  analyzer.load_analysis(save_info=params.save_info)
  analyzer.model_name = params.model_name

for analyzer, params in zip(analyzer_list, params_list):
  print(analyzer.analysis_params.display_name)
  print("Computing the iso-response vectors...")
  outputs = compute_iso_vectors(analyzer, min_angle, max_angle, num_neurons, use_bf_stats)
  analyzer.target_neuron_ids = outputs[0]
  analyzer.comparison_neuron_ids = outputs[1]
  analyzer.target_vectors = outputs[2]
  analyzer.rand_orth_vectors = outputs[3]
  analyzer.comparison_vectors = outputs[4]
  for use_random_orth_vects, rand_str in zip([True, False], "rand, comparison"):
    print("Generating "+rand_str+" dataset...")
    contour_dataset, datapoints = get_contour_dataset(analyzer, num_comparison_vects,
      use_random_orth_vects, x_range, y_range, num_images)
    print("Computing network activations for "+rand_str+" dataset...")
    activations = get_normalized_activations(analyzer, datapoints)
    if use_random_orth_vects:
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_rand_activations_"+params.save_info+".npz",
        data=activations)
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_rand_contour_dataset_"+params.save_info+".npz",
        data=contour_dataset)
    else:
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_comp_activations_"+params.save_info+".npz",
        data=activations)
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_comp_contour_dataset_"+params.save_info+".npz",
        data=contour_dataset)
  params.min_angle = min_angle
  params.max_angle = max_angle
  params.num_neurons = num_neurons
  params.use_bf_stats = use_bf_stats
  params.num_comparison_vects = num_comparison_vects
  params.x_range = x_range
  params.y_range = y_range
  params.num_images = num_images
  np.savez(analyzer.analysis_out_dir+"savefiles/iso_params_"+params.save_info+".npz",
    data=params.__dict__)
