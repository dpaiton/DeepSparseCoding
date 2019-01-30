import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from data.dataset import Dataset
import data.data_selector as ds
import utils.data_processing as dp
import utils.plot_functions as pf
import analysis.analysis_picker as ap
import matplotlib.gridspec as gridspec
import pdb

#List of models for analysis
analysis_list = [
    ##MLP on latent
    #("mlp_lca", "mlp_lca_latent_mnist"),
    #("mlp_sae", "mlp_sae_latent_mnist"),
    #("mlp_lista", "mlp_lista_5_mnist"),
    #("mlp_lista", "mlp_lista_20_mnist"),
    #    #MLP on pixels
        ("mlp_lca", "mlp_lca_recon_mnist"),
        ("mlp_sae", "mlp_sae_recon_mnist"),
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
  [0.0, 0.0, 1.0], #"b"
  [0.0, 1.0, 0.0], #"g"
  [0.0, 1.0, 1.0], #"c"
  [1.0, 0.0, 1.0], #"m"
  [1.0, 1.0, 0.0], #"y"
  [0.0, 0.0, 1.0], #"k"
  ]

title_font_size = 16
axes_font_size = 16

plot_all = True
plot_over_time = False

#Base outdir for multi-network plot
outdir = "/home/slundquist/Work/Projects/vis/"

class params(object):
  def __init__(self):
    self.model_type = ""
    self.model_name = ""
    self.plot_title_name = self.model_name.replace("_", " ").title()
    self.version = "0.0"
    #TODO can we pull this save_info from somewhere?
    #self.save_info = "analysis_carlini"
    self.save_info = "analysis_test_kurakin"
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

#Use this index in the table
#TODO pick best recon mult for here
recon_mult_idx = 0
eval_batch_size = 100

#Store csv file with corresponding accuracy
#Here, x axis of table is target network
#y axis is source network
header_y = [a[1] for a in analysis_list]
header_x = ["clean",] + header_y
output_table = np.zeros((len(header_y), len(header_x)))
#Loop thorugh source network
for source_model_idx, (model_type, model_name) in enumerate(analysis_list):
  analysis_params = params()
  analysis_params.model_type = model_type
  analysis_params.model_name = model_name
  analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
  source_analyzer = setup(analysis_params)
  #Fill in clean accuracy
  output_table[source_model_idx, 0] = source_analyzer.adversarial_clean_accuracy

  #Calculate true classes of provided images
  input_classes = np.argmax(source_analyzer.adversarial_input_labels, axis=-1)

  #Loop through target networks
  for target_model_idx, (model_type, model_name) in enumerate(analysis_list):
    #If source == target, just grab accuracy from analysis
    if(source_model_idx == target_model_idx):
      output_table[source_model_idx, target_model_idx + 1] = source_analyzer.adversarial_adv_accuracy
    else:
      analysis_params = params()
      analysis_params.model_type = model_type
      analysis_params.model_name = model_name
      analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
      target_analyzer = setup(analysis_params)

      #Get adv examples from source
      source_adv_examples = source_analyzer.adversarial_images[recon_mult_idx, -1, ...]
      source_adv_examples = dp.reshape_data(source_adv_examples, target_analyzer.model_params.vectorize_data)[0]

      #Evaluate on target model
      label_est = target_analyzer.evaluate_model_batch(eval_batch_size,
          source_adv_examples, ["label_est:0"])["label_est:0"]
      classes_est = np.argmax(label_est, axis=-1)
      output_table[source_model_idx, target_model_idx + 1] = np.mean(classes_est == input_classes)

#Construct big numpy array to convert to csv
csv_out_array = np.concatenate((np.array(header_y)[:, None], output_table), axis=1)
header_x = ["",] + header_x
csv_out_array = np.concatenate((np.array(header_x)[None, :], csv_out_array), axis=0)
np.savetxt(outdir + "transfer_accuracy.csv", csv_out_array, fmt="%s", delimiter=",")
pdb.set_trace()

fig, ax = plt.subplots()

#TODO only do this analysis if we're sweeping
for model_idx, (model_type, model_name) in enumerate(analysis_list):
  analysis_params = params()
  analysis_params.model_type = model_type
  analysis_params.model_name = model_name
  analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

  analyzer = setup(analysis_params)

  batch_size = analyzer.analysis_params.adversarial_batch_size

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
  recon_mult = np.array(analyzer.analysis_params.recon_mult)
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
ax.set_xscale("log")
ax.set_xlabel("Recon Mult value", fontsize=axes_font_size)
ax.set_ylabel("Input Adv MSE", fontsize=axes_font_size)
ax.legend()

fig.savefig(outdir + "/class_mult_tradeoff.png")

plt.close('all')


pdb.set_trace()



analysis_params = params()
analyzer = setup(analysis_params)

class_adversarial_file_loc = analyzer.analysis_out_dir+"savefiles/class_adversary_"+analysis_params.save_info+".npz"
assert os.path.exists(class_adversarial_file_loc), (class_adversarial_file_loc+" must exist.")

batch_size = analyzer.analysis_params.adversarial_batch_size
orig_imgs = analyzer.class_adversarial_input_images.reshape(
  int(batch_size),
  int(np.sqrt(analyzer.model.num_pixels)),
  int(np.sqrt(analyzer.model.num_pixels)))
for idx in range(batch_size):
  pf.plot_image(orig_imgs[idx], title="Input Image",
    save_filename=analyzer.analysis_out_dir+"/vis/"+analysis_params.save_info+\
    "_adversarial_input_batch_"+str(idx)+".png")

target_classes = np.argmax(analyzer.adversarial_target_labels, axis=-1)

plot_int = 100
for step, (stim, output) in enumerate(zip(analyzer.adversarial_images[0], analyzer.adversarial_outputs[0])):
  if(step % plot_int == 0):
    adv_imgs = stim.reshape(
      int(batch_size),
      int(np.sqrt(analyzer.model.num_pixels)),
      int(np.sqrt(analyzer.model.num_pixels)))
    for idx in range(batch_size):
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

#orig_recon = analyzer.adversarial_recons[0].reshape(
#  int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
#adv_recon = analyzer.adversarial_recons[-1].reshape(
#  int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
#adv_img = analyzer.adversarial_images[-1].reshape(int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
#
##rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#plt.rc('text', usetex=True)
#
#title_font_size = 16
#axes_font_size = 16
#
#fig = plt.figure()
#gs = gridspec.GridSpec(5, 5)
#gs.update(wspace=.5, hspace=1)
#
#ax = plt.subplot(gs[0, 0:3])
#ax = pf.clear_axis(ax)
#ax.text(0.5, 0.5, analysis_params.plot_title_name, fontsize=title_font_size,
#       horizontalalignment='center', verticalalignment='center')
#
#ax = plt.subplot(gs[2, 0])
#ax = pf.clear_axis(ax)
#ax.imshow(target_img, cmap='gray')
#ax.set_title(r"$S_t$", fontsize = title_font_size)
#
#ax = plt.subplot(gs[1, 1])
#ax = pf.clear_axis(ax)
#ax.imshow(orig_img, cmap='gray')
#ax.set_title(r"$S_i$", fontsize = title_font_size)
#
#ax = plt.subplot(gs[2, 1])
#ax = pf.clear_axis(ax)
#ax.imshow(orig_recon, cmap='gray')
#ax.set_title(r"$\hat{S}_i$", fontsize = title_font_size)
#
#ax = plt.subplot(gs[1, 2])
#ax = pf.clear_axis(ax)
#ax.imshow(adv_img, cmap='gray')
#ax.set_title(r"$S^*$", fontsize = title_font_size)
#
#ax = plt.subplot(gs[2, 2])
#ax = pf.clear_axis(ax)
#ax.imshow(adv_recon, cmap='gray')
#ax.set_title(r"$\hat{S}^*$", fontsize = title_font_size)
#
#axbig = plt.subplot(gs[3:, :3])
#
#line1 = axbig.plot(analyzer.adversarial_input_adv_mses, 'r', label="input to perturbed")
#axbig.set_ylim([0, np.max(analyzer.adversarial_input_adv_mses+analyzer.adversarial_target_recon_mses+analyzer.adversarial_target_adv_mses+analyzer.adversarial_adv_recon_mses)])
#axbig.tick_params('y', colors='k')
#axbig.set_xlabel("Step", fontsize=axes_font_size)
#axbig.set_ylabel("MSE", fontsize=axes_font_size)
#axbig.set_ylim([0, np.max(analyzer.adversarial_target_recon_mses+analyzer.adversarial_target_adv_mses+analyzer.adversarial_adv_recon_mses)])
#
##ax2 = ax1.twinx()
#line2 = axbig.plot(analyzer.adversarial_target_adv_mses, 'b', label="target to perturbed")
##ax2.tick_params('y', colors='k')
##ax2.set_ylim(ax1.get_ylim())
#
##ax3 = ax1.twinx()
#line3 = axbig.plot(analyzer.adversarial_target_recon_mses, 'g', label="target to recon")
##ax3.tick_params('y', colors='k')
##ax3.set_ylim(ax1.get_ylim())
#
#line4 = axbig.plot(analyzer.adversarial_adv_recon_mses, 'k', label="perturbed to recon")
#
#lines = line1+line2+line3+line4
##lines = line2+line3+line4
#line_labels = [l.get_label() for l in lines]
#
##Set legend to own ax
#ax = plt.subplot(gs[3, 3])
#ax = pf.clear_axis(ax)
#ax.legend(lines, line_labels, loc='upper left')
#
#fig.savefig(analyzer.analysis_out_dir+"/vis/adversarial_losses.pdf")
##plt.show()
#
