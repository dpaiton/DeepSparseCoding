import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from skimage.measure import compare_psnr
from data.dataset import Dataset
import data.data_selector as ds
import utils.data_processing as dp
import utils.plot_functions as pf
import analysis.analysis_picker as ap
import matplotlib.gridspec as gridspec
import pdb
import seaborn as sn
import pandas as pd

#List of models for analysis
analysis_list = [
    #MLP on latent
    ("mlp_lca", "mlp_lca_768_latent_mnist"),
    ("mlp_lca", "mlp_lca_1568_latent_mnist"),
    ("mlp_lista", "mlp_lista_5_mnist"),
    ("mlp_lista", "mlp_lista_20_mnist"),
    ("mlp_lista", "mlp_lista_50_mnist"),
    #    #MLP on pixels
    ("mlp_lca", "mlp_lca_768_recon_mnist"),
    #("mlp_lca", "mlp_lca_1568_recon_mnist"),
    ("mlp_sae", "mlp_sae_768_recon_mnist"),
    ("mlp", "mlp_adv_mnist"),
    ("mlp", "mlp_mnist"),
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
  [1.0, 0.5, 0.5], #"dark r"
  [0.0, 0.0, 0.5], #"light b"
  [0.0, 0.0, 1.0], #"b"
  [0.5, 0.5, 1.0], #"dark b"

  [0.0, 1.0, 0.0], #"g"
  [0.5, 1.0, 0.5], #"dark g"
  [0.0, 1.0, 1.0], #"c"
  [1.0, 0.0, 1.0], #"m"
  [1.0, 1.0, 0.0], #"y"
  [0.0, 0.0, 1.0], #"k"
  ]

title_font_size = 16
axes_font_size = 16

#save_info = "analysis_test_carlini_targeted"
#TODO pick best recon mult for here
#recon_mult_idx = 4

#save_info = "analysis_test_kurakin_untargeted"
save_info = "analysis_test_kurakin_targeted"
recon_mult_idx = 0

construct_heatmap = False
construct_adv_examples = True
construct_class_mult_tradeoff = False
construct_over_time = False

eval_batch_size = 100
num_output_batches = 3

#Base outdir for multi-network plot
outdir = "/home/slundquist/Work/Projects/vis/"

class params(object):
  def __init__(self):
    self.model_type = ""
    self.model_name = ""
    self.plot_title_name = self.model_name.replace("_", " ").title()
    self.version = "0.0"
    #TODO can we pull this save_info from somewhere?
    self.save_info = save_info
    self.overwrite_analysis_log = False

def makedir(name):
  if not os.path.exists(name):
    os.makedirs(name)

def setup(params):
  params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+params.model_name)
  analyzer = ap.get_analyzer(params.model_type)
  analyzer.setup(params)
  analyzer.model.setup(analyzer.model_params)
  analyzer.load_analysis(save_info=params.save_info)
  makedir(analyzer.analysis_out_dir+"/vis/"+params.save_info+"_adversarial_stims/")
  return analyzer

makedir(outdir)

if(construct_heatmap):
  ###Confusion matrix for transferability
  #Store csv file with corresponding accuracy
  #Here, x axis of table is target network
  #y axis is source network
  header_y = [a[1] for a in analysis_list]
  header_x = header_y[:]
  output_table = np.zeros((len(header_y), len(header_x)))
  #Loop thorugh source network
  for source_model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
    source_analyzer = setup(analysis_params)

    #Fill in clean accuracy
    header_y[source_model_idx] = header_y[source_model_idx] + \
      ": %.2f"%source_analyzer.adversarial_clean_accuracy[recon_mult_idx]

    #output_table[source_model_idx, 0] = source_analyzer.adversarial_clean_accuracy[recon_mult_idx]

    #Calculate true classes of provided images
    input_classes = np.argmax(source_analyzer.adversarial_input_labels, axis=-1)

    #Get adv examples from source
    source_adv_examples = source_analyzer.adversarial_images[recon_mult_idx, -1, ...]

    #Loop through target networks
    for target_model_idx, (model_type, model_name) in enumerate(analysis_list):
      ##If source == target, just grab accuracy from analysis
      #if(source_model_idx == target_model_idx):
      #  output_table[source_model_idx, target_model_idx] = \
      #    source_analyzer.adversarial_adv_accuracy[recon_mult_idx]
      #else:
        analysis_params = params()
        analysis_params.model_type = model_type
        analysis_params.model_name = model_name
        analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
        target_analyzer = setup(analysis_params)

        reshape_source_adv_examples = dp.reshape_data(source_adv_examples, target_analyzer.model_params.vectorize_data)[0]

        #Evaluate on target model
        label_est = target_analyzer.evaluate_model_batch(eval_batch_size,
            reshape_source_adv_examples, ["label_est:0"])["label_est:0"]
        classes_est = np.argmax(label_est, axis=-1)
        output_table[source_model_idx, target_model_idx] = np.mean(classes_est == input_classes)

        #Sanity check, take out after
        if(source_model_idx == target_model_idx):
          assert(output_table[source_model_idx, target_model_idx] == \
            source_analyzer.adversarial_adv_accuracy[recon_mult_idx])

  ##Construct big numpy array to convert to csv
  #csv_out_array = np.concatenate((np.array(header_y)[:, None], output_table), axis=1)
  #header_x = ["",] + header_x
  #csv_out_array = np.concatenate((np.array(header_x)[None, :], csv_out_array), axis=0)
  #np.savetxt(outdir + "transfer_accuracy.csv", csv_out_array, fmt="%s", delimiter=",")

  #Construct heatmap
  df_cm = pd.DataFrame(output_table, index = header_y, columns = header_x)
  fig = plt.figure(figsize = (10, 7))
  sn.heatmap(df_cm, annot=True)
  plt.title(analysis_params.save_info)
  plt.ylabel("Adv Target Models")
  plt.xlabel("Eval Models")
  plt.tight_layout()
  fig.savefig(outdir + "/transfer_accuracy_"+analysis_params.save_info+".png")
  plt.close("all")

###Examples of adv inputs
if construct_adv_examples:
  imgs = []
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
    analyzer = setup(analysis_params)

    num_data = analyzer.num_data

    #Get adv examples from source
    adv_examples = analyzer.adversarial_images[recon_mult_idx, -1, ...].reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    orig_image = analyzer.adversarial_images[recon_mult_idx, 0, ...].reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    output = np.argmax(analyzer.adversarial_outputs[recon_mult_idx, -1], axis=-1)
    pert = adv_examples - orig_image
    imgs.append([pert, adv_examples, output])

  for batch_id in range(num_output_batches):
    #Construct img table
    fig, ax = plt.subplots(len(analysis_list), 4)
    plt.suptitle(analysis_params.save_info)

    ax[0, 1].set_title("pert")
    ax[0, 2].set_title("adv_image")
    ax[0, 3].set_title("output_class")
    for model_idx, (model_type, model_name) in enumerate(analysis_list):
      ax[model_idx, 0].text(1, 0.5, model_name, horizontalalignment="right", verticalalignment="center")
      pert_img = ax[model_idx, 1].imshow(imgs[model_idx][0][batch_id], cmap="gray")

      cb = fig.colorbar(pert_img, ax=ax[model_idx, 1], aspect=5)
      tick_locator = ticker.MaxNLocator(nbins=3)
      cb.locator = tick_locator
      cb.update_ticks()

      ax[model_idx, 2].imshow(imgs[model_idx][1][batch_id], cmap="gray")
      ax[model_idx, 3].text(0.5, 0.5, str(imgs[model_idx][2][batch_id]),
        horizontalalignment="center", verticalalignment="center")

      for i in range(4):
        pf.clear_axis(ax[model_idx, i])

    fig.savefig(outdir+"/adv_class_example_"+analysis_params.save_info+"_batch_" + str(batch_id)+ ".png")
    plt.close("all")

if construct_class_mult_tradeoff:
  fig, ax = plt.subplots()
  #TODO only do this analysis if we're sweeping
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

    analyzer = setup(analysis_params)

    #Grab final mses
    #These mses are in shape [num_recon_mults, num_iterations, num_batch]
    input_adv_vals = np.array(analyzer.adversarial_input_adv_mses)[:, -1, :]

    clean_accuracy = analyzer.adversarial_clean_accuracy

    #TODO this isn't working for mlp_vae
    ##accuracy should be within threshold no matter the recon val being tested
    ##Here, not identical because vae's sample latent space, so output class is non-deterministic
    #try:
    #  assert np.all(np.abs(clean_accuracy[0] - np.mean(clean_accuracy, axis=0)) < 1e-3)
    #except:
    #  pdb.set_trace()

    target_classes = np.argmax(np.array(analyzer.adversarial_target_labels), axis=-1)
    #Grab output from adversarial inputs
    #These are in shape [num_recon_mults, num_iterations, num_batch, num_classes]
    adv_output_classes = np.argmax(np.array(analyzer.adversarial_outputs)[:, -1, :, :], axis=-1)
    attack_accuracy = np.mean((target_classes[None, :] == adv_output_classes).astype(np.float32), axis=-1)
    recon_mult = np.array(analyzer.analysis_params.carlini_recon_mult)
    label_str = model_name + " model_accuracy:"+str(np.mean(clean_accuracy))

    color = colors[model_idx]

    #ax.scatter(recon_mult,
    #  np.mean(input_adv_vals, axis=-1),
    #  label = label_str, c=np.array(all_colors))

    for i in range(recon_mult.shape[0]):
      if(i == 0):
        label = label_str
      else:
        label = None
      marker, cap, bar = ax.errorbar(recon_mult[i],
        np.mean(input_adv_vals[i]),
        yerr = np.std(input_adv_vals[i]),
        label = label, c=color, fmt="o")
      #Set accuracy as alpha channel
      #Accuracy ranging from [.3, 1]
      alpha_val = attack_accuracy[i] * .7 + .3
      marker.set_alpha(alpha_val)
      [b.set_alpha(alpha_val) for b in bar]


  #ax.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
  #ax.set_ylabel("Attack success rate", fontsize=axes_font_size)
  plt.title(analysis_params.save_info)
  ax.set_xscale("log")
  ax.set_xlabel("Recon Mult value", fontsize=axes_font_size)
  ax.set_ylabel("Input Adv MSE", fontsize=axes_font_size)
  ax.legend()

  fig.savefig(outdir + "/class_mult_tradeoff"+analysis_params.save_info+".png")

  plt.close('all')

if construct_over_time:
  #Loop thorugh source network
  for (model_type, model_name) in analysis_list:
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

    analyzer = setup(analysis_params)

    class_adversarial_file_loc = analyzer.analysis_out_dir+"savefiles/class_adversary_"+analysis_params.save_info+".npz"
    assert os.path.exists(class_adversarial_file_loc), (class_adversarial_file_loc+" must exist.")

    num_data = analyzer.num_data
    orig_imgs = analyzer.class_adversarial_input_images.reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))
    for idx in range(num_data):
      pf.plot_image(orig_imgs[idx], title="Input Image",
        save_filename=analyzer.analysis_out_dir+"/vis/"+analysis_params.save_info+\
        "_adversarial_input_batch_"+str(idx)+".png")

    target_classes = np.argmax(analyzer.adversarial_target_labels, axis=-1)
    steps = analyzer.steps_idx

    for (step, stim, output) in zip(steps, analyzer.adversarial_images[0], analyzer.adversarial_outputs[0]):
      adv_imgs = stim.reshape(
        int(num_data),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))
      for idx in range(num_data):
        f, axarr = plt.subplots(2, 1)
        axarr[0].imshow(adv_imgs[idx], cmap='gray')
        axarr[0] = pf.clear_axis(axarr[0])
        axarr[1].bar(list(range(analyzer.model.params.num_classes)), output[idx])
        axarr[1].set_ylim([0, 1])
        mse_val = np.mean((adv_imgs[idx] - orig_imgs[idx]) ** 2)
        output_class = np.argmax(output[idx])
        target_class = target_classes[idx]
        axarr[0].set_title("output_class:"+str(output_class) + "  target_class:"+str(target_class)+"  mse:" + str(mse_val))
        f.savefig(analyzer.analysis_out_dir+"/vis/"+analysis_params.save_info+"_adversarial_stims/"
          +"stim_batch_"+str(idx)+"_step_"+str(step)+".png")
        plt.close('all')

