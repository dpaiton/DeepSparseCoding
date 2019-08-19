# In[1]:


import os


# In[2]:


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import axes_grid1
from scipy.ndimage import imread as imread
import tensorflow as tf
import data.data_selector as ds
import utils.data_processing as dp
import utils.plot_functions as pf
import analysis.analysis_picker as ap
import pdb


# In[3]:

data_dir = "/home/slundquist/Work/Datasets/"


class Params(object):
  def __init__(self):
    #self.model_type = "lca_conv"
    #self.model_name = "lca_conv_cifar10"
    self.model_type = "lca"
    self.model_name = "lca_1568_cifar10_gray"
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


out_dir = analyzer.analysis_out_dir+"/vis/"

#Variable is [num_images, num_steps, y, x, neuron]
u = analyzer.inference_stats['u']

if(len(u.shape) == 5):
  [num_images, num_steps, ny, nx, nn] = u.shape
  u = u[:, :, int(ny/2), int(nx/2), :]
else:
  [num_images, num_steps, nn] = u.shape

num_plot_side_y = int(np.round(np.sqrt(nn)))
num_plot_side_x = int(np.ceil(nn/num_plot_side_y))

fig, axarr = plt.subplots(num_plot_side_y, num_plot_side_x, figsize=(100, 100), sharex=True, sharey=True)

for i_n in range(nn):
  i_xn = i_n % num_plot_side_y
  i_yn = int(i_n / num_plot_side_y)
  data = u[0, :, i_n]
  axarr[i_xn, i_yn].plot(data, linewidth=1)

plt.savefig(out_dir + "u_over_time.pdf")
plt.close("all")





# In[ ]:
fig = plot_inference_stats(analyzer.inference_stats, title="Loss During Inference",
    save_filename = out_dir + "loss_during_inference.png")

act_indicator_threshold = 0.80
inf_trace_fig = pf.plot_inference_traces(analyzer.inference_stats, analyzer.model_schedule[0]["sparse_mult"], act_indicator_threshold=act_indicator_threshold)

inf_trace_fig.savefig(analyzer.analysis_out_dir+"/vis/"+analysis_params.model_name+"_inference_traces_dot_thresh-"+str(act_indicator_threshold)+"_"+analysis_params.save_info+".pdf", transparent=True, bbox_inches="tight", pad=0.1)




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

