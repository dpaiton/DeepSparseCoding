import os
import numpy as np
import utils.log_parser as lp
import utils.plot_functions as pf
import models.model_picker as mp

class Analyzer(object):
  def __init__(self, params):
    self.log_file = (params["model_dir"]+"/logfiles/"+params["model_name"]
      +"_v"+params["version"]+".log")
    self.log_text = lp.load_file(self.log_file)
    self.model_params = lp.read_params(self.log_text)
    assert self.model_params["model_type"] == params["model_type"]
    self.model_params["rand_state"] = np.random.RandomState(
      self.model_params["rand_seed"])
    self.model_schedule = lp.read_schedule(self.log_text)

    ## Determine which checkpoint to load
    if "batch_index" in  params:
      assert params["batch_index"] > 0
      self.batch_index = params["batch_index"]
    else:
      batch_idx = 0
      for schedule in self.model_schedule:
        batch_idx += schedule["num_batches"]
      self.batch_index = batch_idx

    self.load_params(params)
    self.model_params["out_dir"] = self.out_dir
    self.model_params["data_dir"] = params["data_dir"]
    self.make_dirs()
    self.load_model()
    self.model.log_params(params)

  """Load analysis parameters into object"""
  def load_params(self, params):
    # Model details
    self.version = params["version"]
    self.model_name = params["model_name"]
    # Analysis details
    self.eval_train = params["eval_train"]
    self.eval_test = params["eval_test"]
    self.eval_val = params["eval_val"]
    # Output details
    self.file_ext = params["file_ext"]
    self.device = params["device"]
    self.out_dir = params["model_dir"]+"/analysis/"+self.version+"/"
    self.cp_loc = (params["model_dir"]+"/checkpoints/"+params["model_name"]
      +"_v"+params["version"]+"_full-"+str(self.batch_index))

  """Make output directories"""
  def make_dirs(self):
    if not os.path.exists(self.out_dir):
      os.makedirs(self.out_dir)

  def load_model(self):
    self.model = mp.get_model(self.model_params["model_type"],
      self.model_params, self.model_schedule)

  """Save a plot of statistics from the log file"""
  def save_log_stats(self):
    stats = lp.read_stats(self.log_text)
    loss_filename = self.out_dir+"log_stats_v"+self.version+self.file_ext
    pf.save_stats(data=stats, labels=None, out_filename=loss_filename)

  def evaluate_model(self, images):
    pass

  #"""
  #plot activity triggered averages
  #"""
  #def save_atas(self, data, datatype):
  #  fig_title = "Activity triggered averages on "+datatype+" data"
  #  for layer_idx, layer in enumerate(data):
  #    ata_filename = (self.out_dir+"act_trig_avg_layer_"+str(layer_idx)+"_"
  #      +datatype+"_v"+self.version+self.file_ext)
  #    ata = layer.reshape(layer.shape[0], int(np.sqrt(
  #      layer.shape[1])), int(np.sqrt(layer.shape[1])))
  #    pf.save_data_tiled(ata, normalize=True, title=fig_title,
  #      save_filename=ata_filename, vmin=-1.0, vmax=1.0)
