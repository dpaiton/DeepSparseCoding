# In[1]:

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import axes_grid1
from scipy.ndimage import imread as imread
import tensorflow as tf
import pdb

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)
    
import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap


# In[3]:

data_dir = "/home/slundquist/Work/Datasets/"


class Params(object):
  def __init__(self):
    self.model_type = "lca_conv"
    self.model_name = "lca_conv_cifar10"
    self.version = "0.0"
    self.save_info = "analysis_train"
    self.data_dir = data_dir
    self.overwrite_analysis_log = False

analysis_params = Params()

# Computed params
analysis_params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"
  +analysis_params.model_name)


# In[4]:


analyzer = ap.get_analyzer(analysis_params.model_type)
analyzer.setup(analysis_params)
analyzer.model.setup(analyzer.model_params)
analyzer.load_analysis(save_info=analysis_params.save_info)


def plot_inference_stats(data, title="", save_filename=None):
  """
  Plot loss values during LCA inference
  Inputs:
    data: [dict] that must contain the "losses"
      this can be created by using the LCA analyzer objects
  """
  #Calculate nnz
  #act is in
  act = data["a"]
  pdb.set_trace()

  labels = [key for key in data["losses"].keys()]
  losses = [val for val in data["losses"].values()]
  num_im, num_steps = losses[0].shape
  means = [None,]*len(labels)
  sems = [None,]*len(labels)
  for loss_id, loss in enumerate(losses):
    means[loss_id] = np.mean(loss, axis=0) # mean across num_imgs
    sems[loss_id] = np.std(loss, axis=0) / np.sqrt(num_im)
  num_plots_y = np.int32(np.ceil(np.sqrt(len(labels))))+1
  num_plots_x = np.int32(np.ceil(np.sqrt(len(labels))))
  gs = gridspec.GridSpec(num_plots_y, num_plots_x)
  fig = plt.figure(figsize=(10,10))
  loss_id = 0
  for plot_id in np.ndindex((num_plots_y, num_plots_x)):
    (y_id, x_id) = plot_id
    ax = fig.add_subplot(gs[plot_id])
    if loss_id < len(labels):
      time_steps = np.arange(num_steps)
      ax.plot(time_steps, means[loss_id], "k-")
      ax.fill_between(time_steps, means[loss_id]-sems[loss_id],
        means[loss_id]+sems[loss_id], alpha=0.2)
      ax.set_ylabel(labels[loss_id].replace('_', ' '), fontsize=16)
      ax.set_xlim([1, np.max(time_steps)])
      ax.set_xticks([1, int(np.floor(np.max(time_steps)/2)), np.max(time_steps)])
      ax.set_xlabel("Time Step", fontsize=16)
      ax.tick_params("both", labelsize=14)
      loss_id += 1
    else:
      ax = pf.clear_axis(ax, spines="none")
  fig.tight_layout()
  fig.suptitle(title, y=1.03, x=0.5, fontsize=20)
  if save_filename is not None:
    fig.savefig(save_filename, transparent=True)
    plt.close(fig)
    return None
  #plt.show()
  return fig


# In[ ]:
out_dir = analyzer.analysis_out_dir+"/vis/"
fig = plot_inference_stats(analyzer.inference_stats, title="Loss During Inference",
    save_filename = out_dir + "loss_during_inference.png")


## In[ ]:
#
#
#num_pixels, num_neurons = analyzer.atas.shape
#atas_fig = pf.plot_data_tiled(analyzer.atas.T.reshape(num_neurons,
#  int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))), normalize=False,
#  title="Activity triggered averages on image data")
#
#
## In[ ]:
#
#
#noise_images = np.random.standard_normal(data["train"].images.shape)
#noise_evals = analyzer.evaluate_model(noise_images, analyzer.var_names)
#noise_atas = analyzer.compute_atas(noise_evals["inference/activity:0"],
#  noise_images)
#noise_atas_fig = pf.plot_data_tiled(noise_atas.T.reshape(num_neurons,
#  int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))), normalize=False,
#  title="Activity triggered averages on standard normal noise data")
#
#
## In[ ]:


#inference_fig = pf.plot_inference_traces(analyzer.inference_stats, analyzer.model_schedule[0]["sparse_mult"])


# In[ ]:


#fig = plot_inference_stats(analyzer.inference_stats, title="Loss During Inference")

