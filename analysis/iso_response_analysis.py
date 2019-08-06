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
  num_above_min = np.count_nonzero(plot_matrix<min_angle)
  sorted_angle_indices = np.stack(np.unravel_index(np.argsort(plot_matrix.ravel()),
    plot_matrix.shape), axis=1)[num_above_min:, :]
  #orig_min_angle = min_angle
  #orig_max_angle = max_angle
  #vectors = np.argwhere(np.logical_and(plot_matrix < max_angle, plot_matrix > min_angle))
  #num_tries=0
  #while len(set(vectors[:,0])) <= num_neurons:
  #  vectors = np.argwhere(np.logical_and(plot_matrix < max_angle, plot_matrix > min_angle))
  #  if min_angle > 5:
  #    min_angle -= 1
  #  if max_angle < 89:
  #    max_angle += 1
  #  num_tries += 1
  #  if num_tries > 100:
  #    print("Unable to find comparison vectors...")
  #    import IPython; IPython.embed(); raise SystemExit
  #if min_angle < orig_min_angle or max_angle > orig_max_angle:
  #  print("compute_iso_vectors:WARNING:"
  #    +"The provided angle range was too small, the new angle range is [%g, %g]"%(min_angle,
  #    max_angle))
  #angles = [plot_matrix[vectors[idx, 0], vectors[idx, 1]] for idx in range(vectors.shape[0])]
  #sort_indices = np.argsort(angles)
  #vectors = vectors[sort_indices, :]
  target_neuron_ids = []
  comparison_neuron_ids = [] # list of lists [num_targets][num_comparisons_per_target]
  target_vectors = []
  rand_orth_vectors = []
  comparison_vectors = []
  #unique_vectors = list(set(vectors[:,0]))
  #for vector_set_id in range(num_neurons):
  candidate_neurons = np.random.choice(range(analyzer.bf_stats["num_outputs"]),
    num_neurons, replace=False)
  for neuron_idx, target_neuron_id in enumerate(candidate_neurons):
    #vector_id = np.argwhere(vectors[:,0] == unique_vectors[vector_set_id])[0]
    #target_neuron_id = vectors[vector_id, 0].item()
    target_neuron_ids.append(target_neuron_id)
    # Reshape & rescale target vector
    target_vector = analyzer.bf_stats["basis_functions"][target_neuron_id]
    target_vector = target_vector.reshape(analyzer.model_params.num_pixels)

    # NORM ADJUSTMENT
    target_vector = target_vector / np.linalg.norm(target_vector)

    target_vectors.append(target_vector)
    # Build matrix of random orthogonal vectors
    rand_orth_vectors.append(dp.get_rand_orth_vectors(target_vector,
      analyzer.model.params.num_pixels-1))

    # Build matrix of comparison vectors (use all neurons)
    target_neuron_locs = np.argwhere(sorted_angle_indices[:,0] == target_neuron_id)
    low_angle_neuron_ids = np.squeeze(sorted_angle_indices[target_neuron_locs, 1])
    extra_indices = []
    for index in range(analyzer.bf_stats["num_outputs"]):
      if index not in low_angle_neuron_ids:
        if index != target_neuron_id:
          extra_indices.append(index)

    if len(extra_indices) > 0:
      try:
        sub_comparison_neuron_ids = np.concatenate((low_angle_neuron_ids, np.array(extra_indices)))
      except:
        import IPython; IPython.embed(); raise SystemExit
    else:
      sub_comparison_neuron_ids = low_angle_neuron_ids

    if(use_bf_stats):
      sub_comparison_neuron_ids = sub_comparison_neuron_ids[:analyzer.bf_stats["num_outputs"]]
    else:
      sub_comparison_neuron_ids = sub_comparison_neuron_ids[:optimal_stims_dict["num_outputs"]]

    #if(use_bf_stats):
    #  #sub_comparison_neuron_ids = [vectors[vector_id, 1].item()]
    #  sub_comparison_neuron_ids = [sorted_angle_indices[neuron_idx, 1]]
    #  for index in range(analyzer.bf_stats["num_outputs"]):
    #    if index != target_neuron_id and index not in sub_comparison_neuron_ids:
    #      sub_comparison_neuron_ids.append(index)
    #else:
    #  sub_comparison_neuron_ids = [index for index in range(optimal_stims_dict["num_outputs"])
    #    if index != target_neuron_id]

    comparison_vector_matrix = target_vector.T[:,None] # matrix of alternate vectors
    for comparison_neuron_id in sub_comparison_neuron_ids:
      if(comparison_neuron_id != target_neuron_id):
        if(use_bf_stats):
          comparison_vector = analyzer.bf_stats["basis_functions"][comparison_neuron_id]
        else:
          comparison_vector = optimal_stims_dict["basis_functions"][comparison_neuron_id]
        comparison_vector = comparison_vector.reshape(analyzer.model_params.num_pixels)

        # NORM ADJUSTMENT
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

def get_normalized_activations(analyzer, contour_dataset, batch_size=None):
  """
  contour_dataset should have shape [num_target_neurons][num_comparisons_per_target][num_datapoints, datapoint_length]
  output list is shape [num_target_neurons][num_comparisons_per_target][num_datapoints_x, num_datapoints_y]
  """
  activations_list = []
  for target_index, neuron_index in enumerate(analyzer.target_neuron_ids):
    activity_sub_list = []
    for comparison_index, datapoints in enumerate(contour_dataset[target_index]):
      num_images, data_size = datapoints["test"].images.shape
      if batch_size is None:
        batch_size = 1
        for n in range(2, num_images):
          if num_images%n == 0:
            #batch_size = num_images // n # second greatest factor
            batch_size = n
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
    self.display_name = "Sparse Coding 512"
    self.version = "0.0"
    self.save_info = "analysis_train_carlini_targeted"
    self.overwrite_analysis_log = False

class lca_768_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_768_vh"
    self.display_name = "Sparse Coding 768"
    self.version = "0.0"
    self.save_info = "analysis_train_carlini_targeted"
    self.overwrite_analysis_log = False

class lca_1024_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_1024_vh"
    self.display_name = "Sparse Coding 1024"
    self.version = "0.0"
    self.save_info = "analysis_train_carlini_targeted"
    self.overwrite_analysis_log = False

class lca_2560_vh_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_2560_vh"
    self.display_name = "Sparse Coding 2560"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class ae_768_vh_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_768_vh"
    self.display_name = "ReLU Autoencoder 768"
    self.version = "1.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class sae_768_vh_params(object):
  def __init__(self):
    self.model_type = "sae"
    self.model_name = "sae_768_vh"
    self.display_name = "Sparse Autoencoder 768"
    self.version = "1.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class rica_768_vh_params(object):
  def __init__(self):
    self.model_type = "rica"
    self.model_name = "rica_768_vh"
    self.display_name = "Linear Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class lca_768_mnist_params(object):
  def __init__(self):
    self.model_type = "lca"
    self.model_name = "lca_768_mnist"
    self.display_name = "Sparse Coding 768"
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
    self.display_name = "ReLU Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False

class sae_768_mnist_params(object):
  def __init__(self):
    self.model_type = "sae"
    self.model_name = "sae_768_mnist"
    self.display_name = "Sparse Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False

class rica_768_mnist_params(object):
  def __init__(self):
    self.model_type = "rica"
    self.model_name = "rica_768_mnist"
    self.display_name = "Linear Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_train_kurakin_targeted"
    self.overwrite_analysis_log = False

class ae_deep_mnist_params(object):
  def __init__(self):
    self.model_type = "ae"
    self.model_name = "ae_deep_mnist"
    self.display_name = "ReLU Autoencoder 768"
    self.version = "0.0"
    self.save_info = "analysis_test_carlini_targeted"
    self.overwrite_analysis_log = False

print("Loading models...")
min_angle = 10
max_angle = 60
use_bf_stats = True # If false, then use optimal stimulus
batch_size = 100

#num_neurons = 2 # How many neurons to plot
#num_comparison_vects = 512 # How many planes to construct (None is all of them)
#x_range = [-2, 2]
#y_range = [-2, 2]
#num_images = int(50**2)

num_neurons = 100 # How many neurons to plot
num_comparison_vects = 300 # How many planes to construct (None is all of them)
#TODO: Check that this isn't generating the same amount of data regardless of range?
x_range = [1.7, 1.7]#[-2.0, 2.0]
y_range = [-2.0, 2.0]
num_images = int(10**2)

#params_list = [lca_768_mnist_params(), lca_1536_mnist_params()]
params_list = [lca_512_vh_params(), lca_768_vh_params(), lca_1024_vh_params(), lca_2560_vh_params()]
#params_list = [lca_2560_vh_params()]
#params_list = [rica_768_vh_params(), ae_768_vh_params(), sae_768_vh_params()]
#params_list = [rica_768_mnist_params(), ae_768_mnist_params(), sae_768_mnist_params()]
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
  assert len(analyzer.comparison_neuron_ids) == num_neurons, (
    "Incorrect number of comparison vectors")
  for comparison_ids_list in analyzer.comparison_neuron_ids:
    assert len(comparison_ids_list) >= num_comparison_vects, (
      "Not enough comparison vectors.")
  outputs = dict(zip(["target_neuron_ids", "comparison_neuron_ids", "target_vectors",
    "rand_orth_vectors", "comparison_vectors"], outputs))
  np.savez(analyzer.analysis_out_dir+"savefiles/iso_vectors_1d_"+params.save_info+".npz",
    data=outputs)
  for use_random_orth_vects, rand_str in zip([True, False], ["rand", "comparison"]):
    print("Generating "+rand_str+" dataset...")
    contour_dataset, datapoints = get_contour_dataset(analyzer, num_comparison_vects,
      use_random_orth_vects, x_range, y_range, num_images)
    print("Computing network activations for "+rand_str+" dataset...")
    activations = get_normalized_activations(analyzer, datapoints, batch_size)
    if use_random_orth_vects:
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_rand_activations_1d_"+params.save_info+".npz",
        data=activations)
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_rand_contour_dataset_1d_"+params.save_info+".npz",
        data=contour_dataset)
    else:
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_comp_activations_1d_"+params.save_info+".npz",
        data=activations)
      np.savez(analyzer.analysis_out_dir+"savefiles/iso_comp_contour_dataset_1d_"+params.save_info+".npz",
        data=contour_dataset)
  params.min_angle = min_angle
  params.max_angle = max_angle
  params.num_neurons = num_neurons
  params.use_bf_stats = use_bf_stats
  params.num_comparison_vects = num_comparison_vects
  params.x_range = x_range
  params.y_range = y_range
  params.num_images = num_images
  np.savez(analyzer.analysis_out_dir+"savefiles/iso_params_1d_"+params.save_info+".npz",
    data=params.__dict__)
