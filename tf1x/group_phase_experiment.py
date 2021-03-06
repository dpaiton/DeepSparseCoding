import os
import sys

import numpy as np
import tensorflow as tf

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)

from DeepSparseCoding.tf1x.utils.logger import Logger
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.models.model_picker as mp
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap
import DeepSparseCoding.tf1x.data.data_selector as ds
from DeepSparseCoding.tf1x.data.dataset import Dataset

class params(object):
  def __init__(self):
    # If false, append to log file
    self.overwrite_analysis_log = False
    # Load in training run stats from log file
    self.do_run_analysis = False
    # Evaluate model variables (specified in analysis class) on images
    self.do_evals = True
    # Dictionary fitting
    self.do_basis_analysis = False
    # LCA Inference analysis
    self.do_inference = False #TODO: Does not work for lca_subspace
    # Activity triggered averages
    self.do_atas = False #TODO: this can produce outputs that are too big for npz; need to batch?
    # Recon adversarial image analysis
    self.do_recon_adversaries = False # TODO: broken for rica
    #Classification adversarial image analysis
    self.do_class_adversaries = False
    # Find optimal stimulus using gradient methods
    self.do_neuron_visualization = False # adversaries must be False
    # Patchwise image recon
    self.do_full_recon = False
    # Orientation and Cross-Orientation analysis
    self.do_orientation_analysis = False # TODO: broken for ae_deep
    # Reconstructions from individual groups for subspace Sparse Coding
    self.do_group_recons = False
    # How many images to use for analysis, patches are generated from these
    self.num_analysis_images = 150#1000
    self.batch_size = 10
    # Contrasts for orientation experiments
    self.contrasts = [1.0]
    # Phases for orientation experiments
    self.phases = np.linspace(-np.pi, np.pi, 16)
    # Orientations for orientation experiments
    self.orientations = np.linspace(0.0, np.pi, 16)
    # Which neurons to run analysis on
    self.neuron_indices = list(range(60))
    # Rescale inputs
    self.input_scale = 1.0

def get_isa_group_act(s, group_size=4):
  num_groups = s.shape[1]//group_size
  group_acts = []
  for i in range(s.shape[0]):
    groups = np.array(np.array_split(s[i], s.shape[1]//group_size))
    group_act = np.sqrt(np.sum(np.power(groups, 2), axis=1))
    group_acts.append(group_act)
  return np.stack(group_acts)

def orientation_tuning(analyzer, contrasts=[0.5], orientations=[np.pi],
  phases=[np.pi], group_indices=None, diameter=-1, scale=1.0, batch_size=10):
  """
  Performs orientation tuning analysis for given parameters
  Inputs:
    analyzer
    contrasts [list or np.array] all experiments will be run at each contrast
    orientations [list or np.array] code will compute neuron response for each orientation
    phases [list or np.array] the mean and max response will be computed across all phases
    group_indices [list or np.array] the experiments will be run for each neuron index specified
      Setting this to the default, None, will result in performing the experiment on all neurons
    diameter [int] diameter of mask for grating stimulus, -1 indicates full-field stimulus
    scale [float] scale of the stimulus. By default the stimulus is scaled between -1 and 1.
  """
  bf_stats = analyzer.bf_stats
  # Stimulus parameters
  if analyzer.analysis_params.model_type in ["lca_subspace", "ica_subspace"]:
    tot_num_groups = analyzer.model.params.num_groups
    if group_indices is None:
      group_indices = np.arange(tot_num_groups)
    num_groups = len(group_indices)
    if analyzer.analysis_params.model_type == "lca_subspace":
      neuron_indices = [analyzer.model.module.group_ids[group_idx][0] for group_idx in group_indices]
    else:
      neuron_indices = [analyzer.model.group_ids[group_idx][0] for group_idx in group_indices]
  else:
    tot_num_neurons = analyzer.model.params.num_neurons
    neuron_indices = group_indices
  num_pixels = bf_stats["patch_edge_size"]**2
  num_neurons = np.asarray(neuron_indices).size
  num_contrasts = np.asarray(contrasts).size
  num_orientations = np.asarray(orientations).size
  num_phases = np.asarray(phases).size
  # Generate a grating with spatial frequency provided by bf_stats
  grating = lambda neuron_idx,contrast,orientation,phase:dp.generate_grating(
    *dp.get_grating_params(bf_stats, neuron_idx, orientation=orientation,
    phase=phase, contrast=contrast, diameter=diameter))
  best_phases = np.zeros((num_neurons, num_contrasts, num_orientations))
  best_orientations = np.zeros((num_neurons, num_contrasts))
  responses =  np.zeros((num_neurons, num_contrasts, num_orientations, num_phases))
  phase_stims = np.stack([grating(neuron_idx, contrast, orientation, phase)
    for neuron_idx in neuron_indices
    for contrast in contrasts
    for orientation in orientations
    for phase in phases], axis=0) #Array containing all stimulus that can be returned for testing
  phase_stims = {"test": Dataset(phase_stims[:,:,:,None], lbls=None, ignore_lbls=None,
    rand_state=analyzer.rand_state)}
  if not analyzer.model_params.whiten_data:
    analyzer.model_params.whiten_method = None
  phase_stims = analyzer.model.preprocess_dataset(phase_stims,
    params={"whiten_data":analyzer.model_params.whiten_data,
    "whiten_method":analyzer.model_params.whiten_method})
  phase_stims = analyzer.model.reshape_dataset(phase_stims, analyzer.model_params)
  phase_stims["test"].images /= np.max(np.abs(phase_stims["test"].images))
  phase_stims["test"].images *= scale
  num_data = phase_stims["test"].images.shape[0]
  assert num_data % batch_size == 0, (str(num_data)+" "+str(batch_size))
  num_iterations = num_data // batch_size
  activations = []
  for it in range(num_iterations):
    batch_start_idx = int(it * batch_size)
    batch_end_idx = int(np.min([batch_start_idx + batch_size, num_data]))
    batch_images = phase_stims["test"].images[batch_start_idx:batch_end_idx, ...]
    if analyzer.analysis_params.model_type == "ica_subspace":
      activations.append(get_isa_group_act(analyzer.evaluate_tf_tensor(
        tensor=analyzer.model.get_encodings(),
        feed_dict=analyzer.model.get_feed_dict(batch_images, is_test=True))))
    elif analyzer.analysis_params.model_type == "lca_subspace":
      activations.append(analyzer.evaluate_tf_tensor(
        tensor=analyzer.model.get_group_activity(),
        feed_dict=analyzer.model.get_feed_dict(batch_images, is_test=True)))
    elif analyzer.analysis_params.model_type == "ica":
      activations.append(analyzer.evaluate_tf_tensor(
        tensor=analyzer.model.get_encodings(),
        feed_dict=analyzer.model.get_feed_dict(batch_images, is_test=True)))
    else:
      assert False
  activations = np.stack(activations, axis=0) # num_batch, batch_size, num_activations
  activations = activations.reshape(num_data, activations.shape[-1]) # collapse batch dim
  if analyzer.analysis_params.model_type in ["lca_subspace", "ica_subspace"]:
    activations = activations.reshape(responses.shape+(tot_num_groups,))
    for subgroup_idx, group_idx in enumerate(group_indices):
      activity_slice = activations[subgroup_idx, :, :, :, group_idx]
      responses[subgroup_idx, ...] = activity_slice
      for co_idx, contrast in enumerate(contrasts):
        for or_idx, orientation in enumerate(orientations):
          phase_activity = activations[subgroup_idx, co_idx, or_idx, :, group_idx]
          best_phases[subgroup_idx, co_idx, or_idx] = phases[np.argmax(phase_activity)]
        best_orientations[subgroup_idx, co_idx] = orientations[np.argmax(best_phases[subgroup_idx, co_idx, :])]
  else:
    activations = activations.reshape(responses.shape+(tot_num_neurons,))
    for sub_neuron_idx, tot_neuron_idx in enumerate(neuron_indices):
      activity_slice = activations[sub_neuron_idx, :, :, :, tot_neuron_idx]
      responses[sub_neuron_idx, ...] = activity_slice
      for co_idx, contrast in enumerate(contrasts):
        for or_idx, orientation in enumerate(orientations):
          phase_activity = activations[sub_neuron_idx, co_idx, or_idx, :, tot_neuron_idx]
          best_phases[sub_neuron_idx, co_idx, or_idx] = phases[np.argmax(phase_activity)]
        best_orientations[sub_neuron_idx, co_idx] = orientations[np.argmax(best_phases[sub_neuron_idx, co_idx, :])]
  return {"contrasts":contrasts, "orientations":orientations, "phases":phases,
    "neuron_indices":neuron_indices, "responses":responses, "best_orientations":best_orientations,
    "best_phases":best_phases, "group_indices":group_indices}

model_types = ["ica_subspace", "lca_subspace", "ica"]#, "lca"]
model_names = ["ica_subspace_vh", "lca_subspace_vh", "ica_vh"]#, "lca_1280_vh"]
model_versions = ["3", "5x_4_1.0_0.2", "0.0"]#, "5x_0.55"]
model_save_infos = ["analysis_train", "analysis_train", "analysis_train_carlini_targeted"]#, "analysis_train_kurakin_targeted"]

for model_stats in zip(model_types, model_names, model_versions, model_save_infos):
  # Get params, set dirs
  analysis_params = params() # construct object
  analysis_params.model_type = model_stats[0]
  analysis_params.model_name = model_stats[1]
  analysis_params.version = model_stats[2]
  analysis_params.save_info = model_stats[3]
  analysis_params.projects_dir = os.path.expanduser("~")+"/Work/ryan_Projects/"
  analysis_params.model_dir = analysis_params.projects_dir+analysis_params.model_name

  model_log_file = (analysis_params.model_dir+"/logfiles/"+analysis_params.model_name
    +"_v"+analysis_params.version+".log")
  model_logger = Logger(model_log_file, overwrite=False)
  model_log_text = model_logger.load_file()
  model_params = model_logger.read_params(model_log_text)[-1]
  analysis_params.model_type = model_params.model_type

  # Initialize & setup analyzer
  analyzer = ap.get_analyzer(analysis_params.model_type)
  analyzer.setup(analysis_params)

  analysis_params.data_type = analyzer.model_params.data_type
  analyzer.model_params.whiten_data = False

  analyzer.setup_model(analyzer.model_params)
  analyzer.load_analysis(save_info=analysis_params.save_info)
  #analyzer.load_basis_stats(analysis_params.save_info)
  analyzer.model_name = analysis_params.model_name

  ot_grating_responses = orientation_tuning(analyzer, analysis_params.contrasts,
    analysis_params.orientations, analysis_params.phases,
    analysis_params.neuron_indices, scale=analysis_params.input_scale,
    batch_size=analysis_params.batch_size)

  analysis_out_dir = analysis_params.model_dir+"/analysis/"+analysis_params.version+"/"
  np.savez(analysis_out_dir+"savefiles/group_ot_responses_"+analysis_params.save_info+".npz",
    data=ot_grating_responses)

import IPython; IPython.embed()
