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
  #("lca", "lca_mnist"),
  ("vae", "vae_one_layer_overcomplete_mnist"),
  #("vae", "vae_one_layer_undercomplete_mnist"),
  #("vae", "vae_two_layer_mnist"),
  #("vae", "vae_three_layer_mnist"),
  ]

#colors for analysis_list
colors = [
  "r",
  "b",
  "g",
  "c",
  "m",
  "k",
  ]

title_font_size = 16
axes_font_size = 16

plot_all = True

outdir = "/home/slundquist/Work/Projects/vis/"

class params(object):
  #model_type = "sigmoid_autoencoder"
  #model_name = "sigmoid_autoencoder"
  model_type = ""
  model_name = ""
  plot_title_name = model_name.replace("_", " ").title()
  #model_type = "lca"
  #model_name = "lca_mnist"
  version = "0.0"
  save_info = "analysis"
  overwrite_analysis_log = False

def makedir(name):
  if not os.path.exists(name):
    os.makedirs(name)

def setup(params):
  params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+params.model_name)
  analyzer = ap.get_analyzer(params.model_type)
  analyzer.setup(params)
  analyzer.model.setup(analyzer.model_params)
  analyzer.load_analysis(save_info=params.save_info)
  makedir(analyzer.analysis_out_dir+"/vis/"+params.save_info+"_adversarial_recons/")
  makedir(analyzer.analysis_out_dir+"/vis/"+params.save_info+"_adversarial_stims/")
  return analyzer

makedir(outdir)

fig_unnorm, ax_unnorm = plt.subplots()
fig_norm, ax_norm = plt.subplots()

saved_rm = [0 for i in analysis_list]

#TODO only do this analysis if we're sweeping
#Right now, single scalar for recon mult breaks this code
#Work around is to make a one element list in analysis
for idx, (model_type, model_name) in enumerate(analysis_list):
  analysis_params = params()
  analysis_params.model_type = model_type
  analysis_params.model_name = model_name
  analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

  analyzer = setup(analysis_params)

  orig_img = analyzer.adversarial_input_image.reshape(int(np.sqrt(analyzer.model.num_pixels)),
    int(np.sqrt(analyzer.model.num_pixels)))
  target_img = analyzer.adversarial_target_image.reshape(int(np.sqrt(analyzer.model.num_pixels)),
    int(np.sqrt(analyzer.model.num_pixels)))

  #Grab final mses
  input_adv_vals = [v[-1] for v in analyzer.adversarial_input_adv_mses]
  target_recon_vals = [v[-1] for v in analyzer.adversarial_target_recon_mses]
  input_recon_val = [v[0] for v in analyzer.adversarial_input_recon_mses]

  #input_recon_val should be within threshold no matter the mse being tested
  #Here, not identical because vae's sample latent space, so recon is non-deterministic
  try:
    assert(np.abs(input_recon_val[0] - np.mean(input_recon_val)) < 1e-3)
  except:
    pdb.set_trace()

  input_recon_val = np.mean(input_recon_val)

  norm_target_recon_vals = np.array(target_recon_vals) / input_recon_val

  #Normalize target_recon mse by orig_recon mse

  recon_mult = np.array(analyzer.analysis_params.recon_mult)

  #Find lowest l2 distance of the two axes to the 0,0
  l2_dist = np.sqrt(np.array(input_adv_vals) ** 2 + np.array(target_recon_vals) ** 2)
  min_idx = np.argmin(l2_dist)
  saved_rm[idx] = (min_idx, recon_mult[min_idx])

  #plt.scatter(input_adv_vals, target_recon_vals, c=recon_mult)
  label_str = model_name + " recon_mse:%.4f"%input_recon_val
  ax_norm.scatter(input_adv_vals, norm_target_recon_vals, label=label_str, c=colors[idx], s=2)
  ax_unnorm.scatter(input_adv_vals, target_recon_vals, label=label_str, c=colors[idx], s=2)

ax_norm.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
ax_norm.set_ylabel("Normalized Target Recon MSE", fontsize=axes_font_size)
ax_norm.legend()

ax_unnorm.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
ax_unnorm.set_ylabel("Target Recon MSE", fontsize=axes_font_size)
ax_unnorm.legend()

fig_unnorm.savefig(outdir + "/recon_mult_tradeoff.png")
fig_norm.savefig(outdir + "/norm_recon_mult_tradeoff.png")

plt.close('all')

for idx, (model_type, model_name) in enumerate(analysis_list):
  analysis_params = params()
  analysis_params.model_type = model_type
  analysis_params.model_name = model_name
  analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

  analyzer = setup(analysis_params)

  orig_img = analyzer.adversarial_input_image.reshape(int(np.sqrt(analyzer.model.num_pixels)),
    int(np.sqrt(analyzer.model.num_pixels)))
  target_img = analyzer.adversarial_target_image.reshape(int(np.sqrt(analyzer.model.num_pixels)),
    int(np.sqrt(analyzer.model.num_pixels)))

  pf.plot_image(orig_img, title="Input Image",
    save_filename=analyzer.analysis_out_dir+"/vis/"+anaylysis_params.save_info+"_adversarial_input.png")

  pf.plot_image(target_img, title="Input Image",
    save_filename=analyzer.analysis_out_dir+"/vis/"+anaylysis_params.save_info+"_adversarial_target.png")

  plot_int = 100
  recon_mult = analyzer.analysis_params.recon_mult
  if(plot_all):
    rm_list = enumerate(recon_mult)
  else:
    #saved_rm is a tuple of (idx, recon_mult val)
    rm_list = [saved_rm[idx]]

  for i_rm, rm in rm_list:
    rm_str = "%.2f"%rm
    for step, recon in enumerate(analyzer.adversarial_recons[i_rm]):
      if(step % plot_int == 0):
        adv_recon = recon.reshape(int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
        pf.plot_image(adv_recon, title="step_"+str(step),
          save_filename=analyzer.analysis_out_dir+"/vis/"+anaylysis_params.save_info+"_adversarial_recons/recon_step_"+str(step)+"_recon_mult_"+rm_str+".png")

    for step, stim in enumerate(analyzer.adversarial_images[i_rm]):
      if(step % plot_int == 0):
        adv_img = stim.reshape(int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
        pf.plot_image(adv_img, title="step_"+str(step),
          save_filename=analyzer.analysis_out_dir+"/vis/"+anaylysis_params.save_info+"_adversarial_stims/stim_step_"+str(step)+"_recon_mult_"+rm_str+".png")

    out_filename = analyzer.analysis_out_dir+"/vis/adversarial_losses_rm_"+rm_str+".pdf"
    print(out_filename)

    orig_recon = analyzer.adversarial_recons[i_rm][0].reshape(
      int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
    adv_recon = analyzer.adversarial_recons[i_rm][-1].reshape(
      int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
    adv_img = analyzer.adversarial_images[i_rm][-1].reshape(int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))

    fig = plt.figure()
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=.5, hspace=1)

    ax = plt.subplot(gs[0, 0:3])
    ax = pf.clear_axis(ax)
    ax.text(0.5, 0.5, analysis_params.plot_title_name + " recon_mult:"+rm_str,
      fontsize=title_font_size,
      horizontalalignment='center', verticalalignment='center')

    ax = plt.subplot(gs[2, 0])
    ax = pf.clear_axis(ax)
    ax.imshow(target_img, cmap='gray')
    ax.set_title(r"$S_t$", fontsize = title_font_size)

    ax = plt.subplot(gs[1, 1])
    ax = pf.clear_axis(ax)
    ax.imshow(orig_img, cmap='gray')
    ax.set_title(r"$S_i$", fontsize = title_font_size)

    ax = plt.subplot(gs[2, 1])
    ax = pf.clear_axis(ax)
    ax.imshow(orig_recon, cmap='gray')
    ax.set_title(r"$\hat{S}_i$", fontsize = title_font_size)

    ax = plt.subplot(gs[1, 2])
    ax = pf.clear_axis(ax)
    ax.imshow(adv_img, cmap='gray')
    ax.set_title(r"$S^*$", fontsize = title_font_size)

    ax = plt.subplot(gs[2, 2])
    ax = pf.clear_axis(ax)
    ax.imshow(adv_recon, cmap='gray')
    ax.set_title(r"$\hat{S}^*$", fontsize = title_font_size)

    axbig = plt.subplot(gs[3:, :3])

    #Generate x idxs based on length of lines and save int
    line_x = np.arange(0, analyzer.analysis_params.adversarial_num_steps,
      analyzer.analysis_params.adversarial_save_int)

    line1 = axbig.plot(line_x, analyzer.adversarial_input_adv_mses[i_rm], 'r', label="input to perturbed")
    axbig.set_ylim([0, np.max(analyzer.adversarial_input_adv_mses[i_rm]+analyzer.adversarial_target_recon_mses[i_rm]+analyzer.adversarial_target_adv_mses[i_rm]+analyzer.adversarial_adv_recon_mses[i_rm])])
    axbig.tick_params('y', colors='k')
    axbig.set_xlabel("Step", fontsize=axes_font_size)
    axbig.set_ylabel("MSE", fontsize=axes_font_size)
    axbig.set_ylim([0, np.max(analyzer.adversarial_target_recon_mses[i_rm]+analyzer.adversarial_target_adv_mses[i_rm]+analyzer.adversarial_adv_recon_mses[i_rm])])

    #ax2 = ax1.twinx()
    line2 = axbig.plot(line_x, analyzer.adversarial_target_adv_mses[i_rm], 'b', label="target to perturbed")
    #ax2.tick_params('y', colors='k')
    #ax2.set_ylim(ax1.get_ylim())

    #ax3 = ax1.twinx()
    line3 = axbig.plot(line_x, analyzer.adversarial_target_recon_mses[i_rm], 'g', label="target to recon")
    #ax3.tick_params('y', colors='k')
    #ax3.set_ylim(ax1.get_ylim())

    line4 = axbig.plot(line_x, analyzer.adversarial_adv_recon_mses[i_rm], 'k', label="perturbed to recon")

    lines = line1+line2+line3+line4
    #lines = line2+line3+line4
    line_labels = [l.get_label() for l in lines]

    #Set legend to own ax
    ax = plt.subplot(gs[3, 3])
    ax = pf.clear_axis(ax)
    ax.legend(lines, line_labels, loc='upper left')

    fig.savefig(out_filename)
    plt.close("all")
