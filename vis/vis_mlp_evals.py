import matplotlib
matplotlib.use('Agg')
import pdb
import numpy as np
import seaborn as sn
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import analysis.analysis_picker as ap
import os

#List of models for analysis
#TODO add pretty names
#TODO can we do multiple groups?
analysis_list = [
    ("mlp_lca", "mlp_lca_conv_cifar10_0.05_2_layer_load_from_lca_cifar10_patches_0.5"),
    ("mlp_lca", "mlp_lca_conv_cifar10_0.05_2_layer_load_from_lca_cifar10_patches_1.0"),
    ("mlp_lca", "mlp_lca_conv_cifar10_0.05_2_layer_load_from_lca_cifar10_patches_1.5"),
    ("mlp_lca", "mlp_lca_conv_cifar10_0.1_2_layer_load_from_lca_cifar10_patches_0.5"),
    ("mlp_lca", "mlp_lca_conv_cifar10_0.1_2_layer_load_from_lca_cifar10_patches_1.0"),
    ("mlp_lca", "mlp_lca_conv_cifar10_0.1_2_layer_load_from_lca_cifar10_patches_1.5"),
    ("mlp_lca", "mlp_lca_conv_cifar10_0.2_2_layer_load_from_lca_cifar10_patches_0.5"),
    ("mlp_lca", "mlp_lca_conv_cifar10_0.2_2_layer_load_from_lca_cifar10_patches_1.0"),
    #("mlp_lca", "mlp_lca_conv_cifar10_0.2_2_layer_load_from_lca_cifar10_patches_1.5"),
    ]


outdir = "/home/slundquist/Work/Projects/vis/"
save_info = "analysis_test"

class params(object):
  def __init__(self):
    self.model_type = ""
    self.model_name = ""
    self.plot_title_name = self.model_name.replace("_", " ").title()
    self.version = "0.0"
    #TODO can we pull this save_info from somewhere?
    self.save_info = save_info
    self.overwrite_analysis_log = False

def setup(params):
  params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+params.model_name)
  analyzer = ap.get_analyzer(params.model_type)
  analyzer.setup(params)
  analyzer.model.setup(analyzer.model_params)
  analyzer.load_analysis(save_info=params.save_info)
  return analyzer

def makedir(name):
  if not os.path.exists(name):
    os.makedirs(name)

def build_heatmap(table, header_y, header_x, title, label_y, label_x, outfn):
  df = pd.DataFrame(table, index=header_y, columns=header_x)
  #TODO find fig size automatically
  fig = plt.figure(figsize = (10, 7))
  sn.heatmap(df, annot=True, fmt=".4f", mask=df.isnull())
  plt.title(title)
  plt.ylabel(label_y)
  plt.xlabel(label_x)
  plt.yticks(rotation=0)
  plt.xticks(rotation=90)
  plt.tight_layout()
  fig.savefig(outfn)
  plt.close("all")

makedir(outdir)



#Find axes of tables from filenames
model_runs = []
model_loads = []
#Generate lookup from model name to model type
model_type_lookup = {}

for (model_type, model_name) in analysis_list:
  assert("_load_from_" in model_name)
  str_split = model_name.split("_load_from_")
  model_runs.append(str_split[0])
  model_loads.append(str_split[1])
  model_type_lookup[model_name] = model_type

model_loads = np.sort(np.unique(model_loads))
model_runs = np.sort(np.unique(model_runs))

#Constructing recon error, nnz, and accuracy tables
recon_err_table = np.empty((len(model_loads), len(model_runs)))
recon_err_table[:] = np.nan
nnz_table = np.empty((len(model_loads), len(model_runs)))
nnz_table[:] = np.nan
accuracy_table = np.empty((len(model_loads), len(model_runs)))
accuracy_table[:] = np.nan

#Loop through networks
for i, model_load in enumerate(model_loads):
  for j, model_run in enumerate(model_runs):
    #Reconstruct model name
    model_name = model_run + "_load_from_" + model_load

    #Skip if these don't exist
    if(model_name in model_type_lookup):
      model_type = model_type_lookup[model_name]
    else:
      print(model_name + " folder does not exist, skipping")
      continue

    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analyzer = setup(analysis_params)

    #Skip if these don't exist
    try:
      tmp = analyzer.evals
    except:
      print(model_name + " does not contain evals, skipping")
      continue


    #TODO recon err and nnz might not exist
    input_node = analyzer.evals["input_node:0"]
    label_est = analyzer.evals["label_est:0"]
    reconstruction = analyzer.evals["reconstruction:0"]
    activations = analyzer.evals["activations:0"]
    labels = analyzer.evals["labels"]

    #Calculate stats
    recon_err = np.mean(np.sum((input_node - reconstruction) ** 2, axis=(1, 2, 3)))

    num_activations = activations.shape[1] * activations.shape[2] * activations.shape[3]
    nnz = np.mean(np.count_nonzero(activations, axis=(1, 2, 3))/num_activations)

    accuracy = np.mean(np.argmax(label_est, axis=1) == np.argmax(labels, axis=1))

    recon_err_table[i, j] = recon_err
    nnz_table[i, j] = nnz
    accuracy_table[i, j] = accuracy

out_prefix = outdir + "model_evals_"+save_info+"_"
build_heatmap(recon_err_table, model_loads, model_runs, "Reconstruction Error",
    "Load Model", "Run Model", out_prefix + "recon_err.png")
build_heatmap(nnz_table, model_loads, model_runs, "Percentage Active",
    "Load Model", "Run Model", out_prefix + "nnz.png")
build_heatmap(accuracy_table, model_loads, model_runs, "Accuracy",
    "Load Model", "Run Model", out_prefix + "accuracy.png")


##Construct heatmaps
#df_cm = pd.DataFrame(output_table, index = header_y, columns = header_x)
#fig = plt.figure(figsize = (10, 7))
#sn.heatmap(df_cm, annot=True)
#plt.title("Adversarial Transfer")
#plt.ylabel("Adv Target Models")
#plt.xlabel("Eval Models")
#plt.tight_layout()
#fig.savefig(outdir + "/transfer_accuracy_"+analysis_params.save_info+".png")
#plt.close("all")
