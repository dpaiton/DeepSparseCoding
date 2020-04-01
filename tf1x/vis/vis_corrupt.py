import matplotlib
matplotlib.use('Agg')
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
import skimage
import seaborn as sn
import pandas as pd
import pickle

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)

import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap

data_dir = "/home/slundquist/Work/Datasets/"

run_corruptions = True

load_saved = False

corruptions = [
  "brightness",
  "defocus_blur",
  "fog",
  "gaussian_blur",
  "glass_blur",
  "jpeg_compression",
  "motion_blur",
  "saturate",
  "snow",
  "speckle_noise",
  "contrast",
  "elastic_transform",
  "frost",
  "gaussian_noise",
  "impulse_noise",
  "pixelate",
  "shot_noise",
  "spatter",
  "zoom_blur"]


#permutations = [
#  "brightness",
#  "gaussian_blur",
#  "gaussian_noise_2",
#  "gaussian_noise_3",
#  "gaussian_noise",
#  "motion_blur",
#  "rotate",
#  "scale",
#  "shear",
#  "shot_noise_2",
#  "shot_noise_3",
#  "shot_noise",
#  "snow",
#  "spatter",
#  "speckle_noise_2",
#  "speckle_noise_3",
#  "speckle_noise",
#  "tilt",
#  "translate",
#  "zoom_blur",
#]

#List of models for analysis
analysis_list = [
    ##MLP on latent
    #("mlp_lca", "mlp_lca_768_latent_mnist"),
    #("mlp_lca", "mlp_lca_1568_latent_mnist"),
    #("mlp_lista", "mlp_lista_5_mnist"),
    #("mlp_lista", "mlp_lista_20_mnist"),
    #("mlp_lista", "mlp_lista_50_mnist"),
    #    #MLP on pixels
    #("mlp_lca", "mlp_lca_768_recon_mnist"),
    ##("mlp_lca", "mlp_lca_1568_recon_mnist"),
    #("mlp_sae", "mlp_sae_768_recon_mnist"),
    #("mlp", "mlp_adv_mnist"),
    #("mlp", "mlp_mnist"),
    ("mlp", "mlp_cifar10"),
    ("mlp_lca", "mlp_lca_conv_recon_cifar10"),
    ("mlp_lca", "mlp_lca_conv_latent_cifar10"),
    ]

#colors for analysis_list
#colors = [
#  "r",
#  "b",
#  "g",
#  "c",
#  "m",
#  "k",
#  ]

colors = [
  [1.0, 0.0, 0.0], #"r"
  [0.0, 0.0, 1.0], #"b"
  [0.0, 1.0, 0.0], #"g"

  [1.0, 0.5, 0.5], #"dark r"
  [0.0, 0.0, 0.5], #"light b"
  [0.5, 0.5, 1.0], #"dark b"

  [0.5, 1.0, 0.5], #"dark g"
  [0.0, 1.0, 1.0], #"c"
  [1.0, 0.0, 1.0], #"m"
  [1.0, 1.0, 0.0], #"y"
  [0.0, 0.0, 1.0], #"k"
  ]

title_font_size = 16
axes_font_size = 16

construct_accuracy = False

recon_analysis = False
num_vis = 4

sparsity_analysis = True
target_corruption = "gaussian_noise"
sparse_mults = [0.07, 0.1, 0.2, 0.3, 0.4]
target_diff = 5

eval_batch_size = 100

#Base outdir for multi-network plot
outdir = "/home/slundquist/Work/Projects/vis/"

class Params(object):
  def __init__(self):
    self.model_type = ""
    self.model_name = ""
    self.plot_title_name = self.model_name.replace("_", " ").title()
    self.version = "0.0"
    #TODO can we pull this save_info from somewhere?
    self.save_info = ""
    self.data_dir = data_dir
    self.overwrite_analysis_log = False

def makedir(name):
  if not os.path.exists(name):
    os.makedirs(name)

def setup(params):
  params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+params.model_name)
  analyzer = ap.get_analyzer(params.model_type)
  analyzer.setup(params)
  analyzer.model.setup(analyzer.model_params)
  return analyzer

makedir(outdir)

if(construct_accuracy):
  def construct_accuracy_heatmap(modify_names, modify_data_dir):
    ###Confusion matrix for transferability
    #Store csv file with corresponding accuracy
    #Here, x axis of table is target network
    #y axis is source network
    header_y = [a[1] for a in analysis_list]
    header_x = ["clean_test",] + modify_names
    output_table = np.zeros((len(header_y), len(header_x)))

    saved_data = {}
    saved_data["header_y"] = header_y
    saved_data["header_x"] = header_x
    saved_data["est"] = {}

    for model_idx, (model_type, model_name) in enumerate(analysis_list):
      analysis_params = Params()
      analysis_params.model_type = model_type
      analysis_params.model_name = model_name
      analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
      analyzer = setup(analysis_params)

      #Get test data
      if(model_idx == 0):
        data = ds.get_data(analyzer.model_params)
        clean_images = data["test"].images
        clean_labels = data["test"].labels

        #Calculate true classes of provided images
        clean_classes = np.argmax(clean_labels, axis=-1)

      #Evaluate model on clean images
      est = analyzer.evaluate_model_batch(eval_batch_size,
        clean_images, ["label_est:0"])["label_est:0"]

      #Calculate accuracy
      est_classes = np.argmax(est, axis=-1)
      clean_accuracy = np.mean(clean_classes == est_classes)
      output_table[model_idx, 0] = clean_accuracy

      if(model_idx == 0):
        #Load labels for modified images (might be different than pure cifar test set)
        modify_classes = np.load(modify_data_dir + "/labels.npy")
        saved_data["modify_classes"] = modify_classes

      saved_data["est"][model_name] = {}
      #Loop through various modifications
      for modify_idx, modify_str  in enumerate(modify_names):
        data_fn = modify_data_dir + modify_str + ".npy"
        modify_images = np.load(data_fn).astype(np.float32)/255
        est = analyzer.evaluate_model_batch(eval_batch_size,
          modify_images, ["label_est:0"])["label_est:0"]
        est_classes = np.argmax(est, axis=-1)
        saved_data["est"][model_name][modify_str] = est

        accuracy = np.mean(est_classes == modify_classes)
        output_table[model_idx, modify_idx+1] = accuracy

    saved_data["output_table"] = output_table

    return saved_data

  def plot_corruption_level_data(saved_data, modify_type):
    est_classes = saved_data["est"]
    modify_classes = saved_data["modify_classes"]
    single_modify_classes = modify_classes[:10000]
    #Sanity check
    for i in range(5):
      assert(np.all(single_modify_classes == modify_classes[i*10000:(i+1)*10000]))

    clean_accuracy = saved_data["output_table"][:, 0]
    for modify_idx, modify_str  in enumerate(modify_names):
      fig = plt.figure(figsize= (10, 7))
      for model_idx, (model_type, model_name) in enumerate(analysis_list):
        accuracy_line = [clean_accuracy[model_idx]]
        modify_est_classes = est_classes[model_name][modify_str]
        for diff_idx in range(5):
          diff_est_classes = np.argmax(modify_est_classes[diff_idx*10000:(diff_idx+1)*10000], axis=-1)
          accuracy_line.append(np.mean(single_modify_classes == diff_est_classes))

        plt.plot(np.arange(len(accuracy_line)), accuracy_line, color=colors[model_idx],
          marker="o", label=model_name)
      plt.ylabel("Accuracy")
      plt.xlabel("Difficulty")
      plt.title("Accuracy on " + modify_str)
      plt.legend()
      plt.tight_layout()
      fig.savefig(outdir + "/corruption_" + modify_str.lower()+"_accuracy.png")
      plt.close("all")

  def plot_data(saved_data, modify_type):
    output_table = saved_data["output_table"]

    header_y = saved_data["header_y"]
    header_x = saved_data["header_x"]

    #Construct bar plot
    bar_width = .25
    #Set positions of bar on x axis
    base_pos = [np.arange(len(header_x))]
    for i in range(len(analysis_list)):
      base_pos.append([x+bar_width for x in base_pos[-1]])

    #Make plot
    fig = plt.figure(figsize = (10, 7))
    for i in range(len(analysis_list)):
      plt.bar(base_pos[i], output_table[i, :], color=colors[i], width=bar_width,
        edgecolor="white", label=header_y[i])

    #Add xticks
    plt.xticks([r + bar_width for r in range(len(output_table[0, :]))],
      header_x, rotation='vertical')
    plt.ylabel("Accuracy")
    plt.xlabel(modify_type + " Type")
    plt.title("Accuracy on " + modify_type)

    #Add legend
    plt.legend()

    ##Construct heatmap
    #df_cm = pd.DataFrame(output_table, index = header_y, columns = header_x)
    #fig = plt.figure(figsize = (10, 7))
    #sn.heatmap(df_cm, annot=True)
    #plt.title("Accuracy on " + modify_type)
    #plt.ylabel("Models")
    #plt.xlabel(modify_type + " Type")
    plt.tight_layout()

    fig.savefig(outdir + "/" + modify_type.lower()+"_accuracy.png")
    plt.close("all")


  modify_data_dir = data_dir + "/CIFAR-10-C/"
  modify_names = corruptions
  modify_type = "Corruption"

  save_fn = outdir + "/" + modify_type.lower() + "_saved_data.pickle"
  if os.path.isfile(save_fn) and load_saved:
    with open(save_fn, "rb") as pickle_in:
      saved_data = pickle.load(pickle_in)
  else:
    saved_data = construct_accuracy_heatmap(modify_names, modify_data_dir)
    with open(save_fn, "wb") as pickle_out:
      pickle.dump(saved_data, pickle_out)

  #plot_data(saved_data, modify_type)
  plot_corruption_level_data(saved_data, modify_type)

if recon_analysis:
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    #Only do models with recons
    if "lca" not in model_name:
      continue

    analysis_params = Params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
    analyzer = setup(analysis_params)

    fig, ax = plt.subplots(len(corruptions), 2*num_vis+1, figsize=(4*num_vis, 2*len(corruptions)))

    for modify_idx, modify_str  in enumerate(corruptions):
      data_fn = data_dir + "/CIFAR-10-C/" + modify_str + ".npy"
      modify_images = np.load(data_fn)[40000:40000+num_vis]
      modify_images = modify_images.astype(np.float32)/255
      #Compute recons of modify_images
      recons = analyzer.evaluate_model_batch(num_vis, modify_images,
        ["lca_conv/output/reconstruction:0"])["lca_conv/output/reconstruction:0"]
      pf.clear_axis(ax[modify_idx, 0])
      ax[modify_idx, 0].text(0.5, 0.5, modify_str,
        horizontalalignment="center", verticalalignment="center")

      #Visualize orig image and recon
      for vis_idx in range(num_vis):
        input_range = [modify_images[vis_idx].min(), modify_images[vis_idx].max()]
        input_image = (modify_images[vis_idx] - input_range[0])/(input_range[1]-input_range[0])
        recon_range = [recons[vis_idx].min(), recons[vis_idx].max()]
        recon_image = (recons[vis_idx] - recon_range[0])/(recon_range[1]-recon_range[0])

        #input_image = skimage.transform.rescale(input_image, 4, order=0, clip=False, preserve_range=True, anti_aliasing=False)
        #recon_image = skimage.transform.rescale(recon_image, 4, order=0, clip=False, preserve_range=True, anti_aliasing=False)

        pf.clear_axis(ax[modify_idx, vis_idx*2+1])
        ax[modify_idx, vis_idx*2+1].imshow(input_image)
        ax[modify_idx, vis_idx*2+1].set_title("Orig [%4.2f, %4.2f]"%(input_range[0], input_range[1]))

        pf.clear_axis(ax[modify_idx, vis_idx*2+2])
        ax[modify_idx, vis_idx*2+2].imshow(recon_image)
        ax[modify_idx, vis_idx*2+2].set_title("Recon [%4.2f, %4.2f]"%(recon_range[0], recon_range[1]))
    plt.tight_layout()
    fig.savefig(outdir+"/corruption_recons_"+model_name+"_2.png")
    plt.close("all")

if sparsity_analysis:
  #target_diff = [1-5]
  def test_sparsity_analysis(sparse_mults, target_corruption, target_diff):
    modify_data_dir = data_dir + "/CIFAR-10-C/"
    saved_data = {}

    modify_classes = np.load(modify_data_dir + "/labels.npy")[:10000]
    saved_data["modify_classes"] = modify_classes

    saved_data["est"] = {}
    saved_data["accuracy"] = {}
    saved_data["nnz"] = {}

    for model_idx, (model_type, model_name) in enumerate(analysis_list):
      #Only do models with recons
      if "lca" not in model_name:
        continue

      analysis_params = Params()
      analysis_params.model_type = model_type
      analysis_params.model_name = model_name
      analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
      analyzer = setup(analysis_params)
      saved_data["est"][model_name] = []
      saved_data["accuracy"][model_name] = []
      saved_data["nnz"][model_name] = []

      for sparse_mult in sparse_mults:
        #Set target sparse mult
        analyzer.model_params.schedule[0]["sparse_mult"] = sparse_mult

        data_fn = modify_data_dir + target_corruption + ".npy"
        modify_images = np.load(data_fn).astype(np.float32)/255
        modify_images = modify_images[(target_diff-1)*10000:target_diff*10000]

        out = analyzer.evaluate_model_batch(eval_batch_size,
          modify_images, ["label_est:0", "lca_conv/inference/activity:0"])

        est = out["label_est:0"]
        act = out["lca_conv/inference/activity:0"]
        num_n = 1
        for s in act.shape:
          num_n *= s
        nnz = np.count_nonzero(act) / num_n
        saved_data["nnz"][model_name].append(nnz)
        saved_data["est"][model_name].append(est)

        est_classes = np.argmax(est, axis=-1)
        accuracy = np.mean(est_classes == modify_classes)
        saved_data["accuracy"][model_name].append(accuracy)
    return saved_data

  def plot_sparse_analysis(saved_data, target_corruption, target_diff):
    fig = plt.figure(figsize = (10, 7))
    for model_idx, (model_type, model_name) in enumerate(analysis_list):
      #Only do models with recons
      if "lca" not in model_name:
        continue
      nnz = saved_data["nnz"][model_name]
      accuracy = saved_data["accuracy"][model_name]
      plt.scatter(nnz, accuracy, c=[colors[model_idx]], label=model_name)
    plt.ylabel("Accuracy")
    plt.xlabel("Fraction Active")
    plt.legend()
    plt.title(target_corruption + " Difficulty " + str(target_diff) + " Accuracy vs Fraction Active")
    plt.tight_layout()
    fig.savefig(outdir + "/" + target_corruption+"_"+str(target_diff) + "_accuracy.png")
    plt.close("all")


  save_fn = outdir + "/corruption_sparsity_test_saved_data.pickle"
  if os.path.isfile(save_fn) and load_saved:
    with open(save_fn, "rb") as pickle_in:
      saved_data = pickle.load(pickle_in)
  else:
    saved_data = test_sparsity_analysis(sparse_mults, target_corruption, target_diff)
    with open(save_fn, "wb") as pickle_out:
      pickle.dump(saved_data, pickle_out)

  plot_sparse_analysis(saved_data, target_corruption, target_diff)

  #plot_data(saved_data, modify_type)
  #plot_corruption_level_data(saved_data, modify_type)






