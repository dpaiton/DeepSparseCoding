import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from skimage.measure import compare_psnr
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
    ("mlp_lca", "mlp_lca_latent_cifar10_gray_2layer"),
    ("mlp_lca", "mlp_lca_latent_cifar10_gray_3layer"),
    ("mlp", "mlp_cifar10_gray_2layer"),
    ("mlp", "mlp_cifar10_gray_3layer"),
    ]


#bar_groups = None
bar_groups = [[0, 1], [2, 3]]
inner_group_names = ["w/ LCA", "w/o LCA"]
outer_group_names = ["2 layers", "3 layers"]

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
  [1.0, 0.5, 0.5], #"dark r"
  [0.0, 0.0, 0.5], #"light b"
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
#recon_mult_idx = 0

#save_info = "analysis_test_kurakin_untargeted"
save_info = "analysis_test_kurakin_targeted"
recon_mult_idx = 0

construct_conf_control = True
use_conf_idx = True

construct_heatmap = False
construct_adv_examples = False
construct_class_mult_tradeoff = False
construct_over_time = True

plot_num_over_time = 4


conf_thresh = .9

eval_batch_size = 100
num_output_batches = 16

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

if(construct_conf_control):
  #saved_info is a list of length num_models, with an inner list saving
  #[target_adv_mses, num_failed]
  conf_saved_info = []
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
    analyzer = setup(analysis_params)

    target_labels = np.argmax(analyzer.adversarial_target_labels, axis=-1)
    #adv_output is in [time, batch, classes]
    adv_output = analyzer.adversarial_outputs[recon_mult_idx]
    input_adv_mses = analyzer.adversarial_input_adv_mses[recon_mult_idx]

    num_time, num_batch, num_class = adv_output.shape


    target_adv_conf = adv_output[:, np.arange(num_batch), target_labels]

    #Get index of first timestep that reaches across threshold
    #Note that argmax here stops at first occurance
    target_conf_idx = np.argmax(target_adv_conf >= conf_thresh, axis=0)

    #Mark number of failed attacks
    num_failed = np.nonzero(target_conf_idx == 0)[0].shape[0]

    #Save the indices of succesful attacks
    success_idx = np.nonzero(target_conf_idx)

    #Get mse from target_conf_idx
    target_adv_mses = input_adv_mses[target_conf_idx, np.arange(num_batch)]

    #Pull out only successful ones
    target_adv_mses = target_adv_mses[success_idx]

    conf_saved_info.append([target_adv_mses, num_failed, target_conf_idx])

  x_label_ticks = []
  vals = []
  errs = []
  color = []
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    target_adv_mses, num_failed, success_idx = conf_saved_info[model_idx]
    vals.append(np.mean(target_adv_mses))
    errs.append(np.std(target_adv_mses))
    color.append(colors[model_idx])
    x_label_ticks.append(model_name+":"+str(num_failed))

  #Set position of bar on X axis
  barWidth = .4
  num_groups = len(bar_groups)
  num_per_group = len(bar_groups[0])

  bar_groups = np.array(bar_groups)

  #Plot bar of ave mse per model
  fig, ax = plt.subplots()
  for i_g in range(num_per_group):
    group_vals = [vals[i] for i in bar_groups[:, i_g]]
    group_err = [errs[i] for i in bar_groups[:, i_g]]
    x_pos = np.arange(num_groups) + i_g*barWidth
    ax.bar(x_pos, group_vals, width=barWidth, yerr=group_err, label=inner_group_names[i_g], color=colors[i_g])

  plt.xlabel('Layers')
  plt.xticks([r + (barWidth)/2 for r in range(num_groups)], outer_group_names)
  plt.legend()


  #ax.bar(np.arange(len(analysis_list)), vals, yerr=errs,
  #  align='center', tick_label=x_label_ticks, color=color)

  #plt.xticks(rotation='vertical')

  #ax.set_xlabel("Model")
  ax.set_ylabel("Input Adv MSE")
  ax.set_title("Average MSE at confidence level "+str(conf_thresh))
  #plt.tight_layout()

  fig.savefig(outdir + "/conf_control_mse_" + analysis_params.save_info+".png")
  plt.close("all")


if(use_conf_idx):
  conf_step_idx = []
  for model_idx in range(len(analysis_list)):
    target_adv_mses, num_failed, step_idx = conf_saved_info[model_idx]
    #Failed attacks have step_idx at 0, replace with -1 to take last step
    step_idx[np.nonzero(step_idx == 0)] = -1
    conf_step_idx.append(step_idx)
else:
  conf_step_idx = [-1 for i in analysis_list]

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
    num_batch = source_analyzer.adversarial_images.shape[2]
    source_adv_examples = source_analyzer.adversarial_images[
      recon_mult_idx, conf_step_idx[source_model_idx], np.arange(num_batch), ...]

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
          try:
            assert(np.abs(output_table[source_model_idx, target_model_idx] -
              source_analyzer.adversarial_adv_accuracy[recon_mult_idx]) <= 0.01)
          except:
            pdb.set_trace()

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


#TODO this doesn't make sense with use_conf_idx
if construct_class_mult_tradeoff:
  assert(not use_conf_idx)
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
    num_batch = analyzer.adversarial_input_adv_mses.shape[2]
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

    #orig_imgs = analyzer.class_adversarial_input_images.reshape(
    #  int(num_data),
    #  int(np.sqrt(analyzer.model.params.num_pixels)),
    #  int(np.sqrt(analyzer.model.params.num_pixels)))
    orig_imgs = analyzer.class_adversarial_input_images

    #for idx in range(num_data):
    #  pf.plot_image(orig_imgs[idx], title="Input Image",
    #    save_filename=analyzer.analysis_out_dir+"/vis/"+analysis_params.save_info+\
    #    "_adversarial_input_batch_"+str(idx)+".png")

    #TODO plot loss over time

    target_classes = np.argmax(analyzer.adversarial_target_labels, axis=-1)
    steps = analyzer.steps_idx
    adv_mses = analyzer.adversarial_input_adv_mses[0, :, :]

    for (step, stim, output, mse_vals) in zip(steps, analyzer.adversarial_images[0], analyzer.adversarial_outputs[0], adv_mses):
      #adv_imgs = stim.reshape(
      #  int(num_data),
      #  int(np.sqrt(analyzer.model.params.num_pixels)),
      #  int(np.sqrt(analyzer.model.params.num_pixels)))
      adv_imgs = stim
      if(adv_imgs.shape[-1] == 1):
        adv_imgs = adv_imgs[:, :, :, 0]

      adv_mses = analyzer.adversarial_input_adv_mses[0, :, :]

      for idx in range(plot_num_over_time):
        f, axarr = plt.subplots(2, 1)
        axarr[0].imshow(adv_imgs[idx], cmap='gray')
        axarr[0] = pf.clear_axis(axarr[0])
        axarr[1].bar(list(range(analyzer.model.params.num_classes)), output[idx])
        axarr[1].set_ylim([0, 1])

        output_class = np.argmax(output[idx])
        target_class = target_classes[idx]

        if("_targeted" in save_info):
          axarr[0].set_title("output_class:"+str(output_class) + "  target_class:"+str(target_class)+"  mse:" + str(mse_vals[idx]))
        else:
          axarr[0].set_title("output_class:"+str(output_class) + "  mse:" + str(mse_vals[idx]))
        f.savefig(analyzer.analysis_out_dir+"/vis/"+analysis_params.save_info+"_adversarial_stims/"
          +"stim_batch_"+str(idx)+"_step_"+str(step)+".png")
        plt.close('all')



#For cifar, TODO make this general
#Idx to string class names
cifar10_class_label_str = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

###Examples of adv inputs
if construct_adv_examples:
  imgs = []
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
    analyzer = setup(analysis_params)

    #Get adv examples from source
    #If last dimension is flat
    #TODO is there a better way to check this?
    if(len(analyzer.adversarial_images.shape) == 4):
      num_data = analyzer.num_data
      adv_examples = analyzer.adversarial_images[recon_mult_idx, conf_step_idx[model_idx], np.arange(num_batch), ...].reshape(
        int(num_data),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))

      orig_image = analyzer.adversarial_images[recon_mult_idx, 0, ...].reshape(
        int(num_data),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))
    else:
      adv_examples = analyzer.adversarial_images[recon_mult_idx, conf_step_idx[model_idx], np.arange(num_batch)]
      orig_image = analyzer.adversarial_images[recon_mult_idx, 0]

    adv_output = np.argmax(
      analyzer.adversarial_outputs[recon_mult_idx, conf_step_idx[model_idx], np.arange(num_batch)], axis=-1)
    orig_output = np.argmax(analyzer.adversarial_outputs[recon_mult_idx, 0], axis=-1)
    pert = adv_examples - orig_image
    imgs.append([pert, adv_examples, orig_output, adv_output])

  for batch_id in range(num_output_batches):
    #Construct img table
    fig, ax = plt.subplots(len(analysis_list)+1, 5)
    plt.suptitle(analysis_params.save_info)

    ax[0, 1].set_title("pert")
    ax[0, 2].set_title("adv_image")
    ax[0, 3].set_title("clean_output")
    ax[0, 4].set_title("adv_output")

    ax[0, 0].text(1, .5, "orig", horizontalalignment="right", verticalalignment="center")

    orig_img_range = [orig_image[batch_id].min(), orig_image[batch_id].max()]
    orig_img = (orig_image[batch_id] - orig_image[batch_id].min())/\
      (orig_image[batch_id].max() - orig_image[batch_id].min())
    ax[0, 2].imshow(orig_img)

    orig_label = np.argmax(analyzer.adversarial_input_labels[batch_id])
    ax[0, 3].text(0.5, 0.5, cifar10_class_label_str[orig_label],
      horizontalalignment="center", verticalalignment="center")

    #Only print out target if attack is targeted
    if("_targeted" in save_info):
      target_label = np.argmax(analyzer.adversarial_target_labels[batch_id])
      ax[0, 4].text(0.5, 0.5, cifar10_class_label_str[target_label],
        horizontalalignment="center", verticalalignment="center")

    for i in range(5):
      pf.clear_axis(ax[0, i])

    for model_idx, (model_type, model_name) in enumerate(analysis_list):
      ax[model_idx+1, 0].text(1, 0.5, model_name, horizontalalignment="right", verticalalignment="center")
      pert_img = imgs[model_idx][0][batch_id]
      adv_img = imgs[model_idx][1][batch_id]
      orig_output = imgs[model_idx][2][batch_id]
      adv_output = imgs[model_idx][3][batch_id]

      pert_range = [pert_img.min(), pert_img.max()]
      pert_img = (pert_img - pert_img.min()) / (pert_img.max() - pert_img.min())
      adv_range = [adv_img.min(), adv_img.max()]
      adv_img = (adv_img - adv_img.min()) / (adv_img.max() - adv_img.min())

      pert_ax = ax[model_idx+1, 1].imshow(pert_img)
      ax[model_idx+1, 1].set_title("[%4.2f , %4.2f]"% (pert_range[0], pert_range[1]))

      #cb = fig.colorbar(pert_img, ax=ax[model_idx, 1], aspect=5)
      #tick_locator = ticker.MaxNLocator(nbins=3)
      #cb.locator = tick_locator
      #cb.update_ticks()

      ax[model_idx+1, 2].imshow(adv_img)
      ax[model_idx+1, 2].set_title("[%4.2f , %4.2f]"% (adv_range[0], adv_range[1]))
      ax[model_idx+1, 3].text(0.5, 0.5, cifar10_class_label_str[orig_output],
        horizontalalignment="center", verticalalignment="center")
      ax[model_idx+1, 4].text(0.5, 0.5, cifar10_class_label_str[adv_output],
        horizontalalignment="center", verticalalignment="center")

      for i in range(5):
        pf.clear_axis(ax[model_idx+1, i])

    fig.savefig(outdir+"/adv_class_example_"+analysis_params.save_info+"_batch_" + str(batch_id)+ ".png")
    plt.close("all")











