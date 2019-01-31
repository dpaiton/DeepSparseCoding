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
  ("lca", "lca_mnist"),
  ("vae", "vae_one_layer_overcomplete_mnist"),
  ("vae", "vae_one_layer_undercomplete_mnist"),
  ("vae", "vae_two_layer_mnist"),
  ("vae", "vae_three_layer_mnist"),
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
plot_over_time = False

#Base outdir for multi-network plot
outdir = "/home/slundquist/Work/Projects/vis/"

class params(object):
  def __init__(self):
    self.model_type = ""
    self.model_name = ""
    self.plot_title_name = model_name.replace("_", " ").title()
    self.version = "0.0"
    self.save_info = "analysis_carlini"
    #self.save_info = "analysis_kurakin"
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
  makedir(analyzer.analysis_out_dir+"/vis/"+params.save_info+"_adversarial_recons/")
  makedir(analyzer.analysis_out_dir+"/vis/"+params.save_info+"_adversarial_stims/")
  return analyzer

makedir(outdir)

fig_unnorm, ax_unnorm = plt.subplots()
fig_norm, ax_norm = plt.subplots()

#saved_rm = [0 for i in analysis_list]

#TODO only do this analysis if we're sweeping
for model_idx, (model_type, model_name) in enumerate(analysis_list):
  analysis_params = params()
  analysis_params.model_type = model_type
  analysis_params.model_name = model_name
  analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

  analyzer = setup(analysis_params)

  batch_size = analyzer.analysis_params.adversarial_batch_size
  orig_img = analyzer.recon_adversarial_input_images.reshape(
    int(batch_size),
    int(np.sqrt(analyzer.model.params.num_pixels)),
    int(np.sqrt(analyzer.model.params.num_pixels)))
  target_img = analyzer.adversarial_target_images.reshape(
    int(batch_size),
    int(np.sqrt(analyzer.model.params.num_pixels)),
    int(np.sqrt(analyzer.model.params.num_pixels)))

  #Grab final mses
  #These mses are in shape [num_recon_mults, num_iterations, num_batch]
  input_adv_vals = np.array(analyzer.adversarial_input_adv_mses)[:, -1, :]
  target_recon_vals = np.array(analyzer.adversarial_target_recon_mses)[:, -1, :]
  input_recon_val = np.array(analyzer.adversarial_input_recon_mses)[:, 0, :]

  #input_recon_val should be within threshold no matter the recon val being tested
  #Here, not identical because vae's sample latent space, so recon is non-deterministic
  try:
    assert np.all(np.abs(input_recon_val[0] - np.mean(input_recon_val, axis=0)) < 1e-3)
  except:
    pdb.set_trace()

  input_recon_val = np.mean(input_recon_val, axis=0)

  norm_target_recon_vals = target_recon_vals / input_recon_val[None, :]

  #Normalize target_recon mse by orig_recon mse
  recon_mult = np.array(analyzer.analysis_params.recon_mult)

  ##Find lowest l2 distance of the two axes to the 0,0
  #l2_dist = np.mean(np.sqrt(np.array(input_adv_vals) ** 2 + np.array(target_recon_vals) ** 2), axis=-1)
  #min_idx = np.argmin(l2_dist)
  #saved_rm[model_idx] = (min_idx, recon_mult[min_idx])

  #plt.scatter(input_adv_vals, target_recon_vals, c=recon_mult)
  label_str = model_name + " recon_mse:%.4f"%np.mean(input_recon_val)
  #TODO how to draw error bars? for now, treat each batch as a scatter data point
  ax_norm.errorbar(np.mean(input_adv_vals, axis=-1),
    np.mean(norm_target_recon_vals, axis=-1),
    xerr = np.std(input_adv_vals, axis=-1),
    yerr = np.std(norm_target_recon_vals, axis=-1),
    label = label_str, c=colors[model_idx], fmt="o")

  ax_unnorm.errorbar(np.mean(input_adv_vals, axis=-1),
    np.mean(target_recon_vals, axis=-1),
    xerr = np.std(input_adv_vals, axis=-1),
    yerr = np.std(target_recon_vals, axis=-1),
    label = label_str, c=colors[model_idx], fmt="o")

  #ax_norm.scatter(input_adv_vals.flatten(), norm_target_recon_vals.flatten(), label=label_str, c=colors[model_idx], s=2)
  #ax_unnorm.scatter(input_adv_vals.flatten(), target_recon_vals.flatten(), label=label_str, c=colors[model_idx], s=2)

ax_norm.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
ax_norm.set_ylabel("Normalized Target Recon MSE", fontsize=axes_font_size)
ax_norm.legend()

ax_unnorm.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
ax_unnorm.set_ylabel("Target Recon MSE", fontsize=axes_font_size)
ax_unnorm.legend()

fig_unnorm.savefig(outdir + "/recon_mult_tradeoff.png")
fig_norm.savefig(outdir + "/norm_recon_mult_tradeoff.png")

plt.close('all')

if(plot_over_time):
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

    analyzer = setup(analysis_params)

    batch_size = analyzer.analysis_params.adversarial_batch_size
    orig_imgs = analyzer.recon_adversarial_input_images.reshape(
      int(batch_size),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))
    target_imgs = analyzer.adversarial_target_images.reshape(
      int(batch_size),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    for batch_idx in range(batch_size):
      pf.plot_image(orig_imgs[batch_idx], title="Input Image",
        save_filename=analyzer.analysis_out_dir+"/vis/"+\
        analysis_params.save_info+"_adversarial_input.png")
      pf.plot_image(target_imgs[batch_idx], title="Target Image",
        save_filename=analyzer.analysis_out_dir+"/vis/"+\
        analysis_params.save_info+"_adversarial_target.png")

    plot_int = 100
    recon_mult = analyzer.analysis_params.recon_mult
    if(plot_all):
      rm_list = enumerate(recon_mult)
    else:
      assert False, ("TODO")
    #else:
    #  #saved_rm is a tuple of (idx, recon_mult val)
    #  rm_list = [saved_rm[model_idx]]

    #plot all recons of adv per step

    for i_rm, rm in rm_list:
      rm_str = "%.2f"%rm
      for step, recon in enumerate(analyzer.adversarial_recons[i_rm]):
        if(step % plot_int == 0):
          adv_recon = recon.reshape(
            int(batch_size),
            int(np.sqrt(analyzer.model.params.num_pixels)),
            int(np.sqrt(analyzer.model.params.num_pixels)))

          for batch_idx in range(batch_size):
            pf.plot_image(adv_recon[batch_idx], title="step_"+str(step),
              save_filename=analyzer.analysis_out_dir+"/vis/"+\
              analysis_params.save_info+"_adversarial_recons/"+\
              "recon_batch_"+str(batch_idx)+"_rm_"+rm_str+"_step_"+str(step)+".png")

      for step, stim in enumerate(analyzer.adversarial_images[i_rm]):
        if(step % plot_int == 0):
          adv_img = stim.reshape(
            int(batch_size),
            int(np.sqrt(analyzer.model.params.num_pixels)),
            int(np.sqrt(analyzer.model.params.num_pixels)))
          for batch_idx in range(batch_size):
            pf.plot_image(adv_img[batch_idx], title="step_"+str(step),
              save_filename=analyzer.analysis_out_dir+"/vis/"+\
              analysis_params.save_info+"_adversarial_stims/"+\
              "stim_batch_"+str(batch_idx)+"_rm_"+rm_str+"_step_"+str(step)+".png")

    for i_rm, rm in rm_list:
      orig_recon = np.array(analyzer.adversarial_recons)[i_rm, 0, ...].reshape(
        int(batch_size),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))

      adv_recon = np.array(analyzer.adversarial_recons)[i_rm, -1, ...].reshape(
        int(batch_size),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))

      adv_img = np.array(analyzer.adversarial_images)[i_rm, -1, ...].reshape(
        int(batch_size),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))

      input_adv_mses = np.array(analyzer.adversarial_input_adv_mses)[i_rm]
      target_recon_mses = np.array(analyzer.adversarial_target_recon_mses)[i_rm]
      target_adv_mses = np.array(analyzer.adversarial_target_adv_mses)[i_rm]
      adv_recon_mses = np.array(analyzer.adversarial_adv_recon_mses)[i_rm]

      for batch_idx in range(batch_size):
        out_filename = analyzer.analysis_out_dir+"/vis/"+\
          "adversarial_losses_"+analysis_params.save_info+"_batch_"+str(batch_idx)+"_rm_"+rm_str+".pdf"
        print(out_filename)

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
        ax.imshow(target_imgs[batch_idx], cmap='gray')
        ax.set_title(r"$S_t$", fontsize = title_font_size)

        ax = plt.subplot(gs[1, 1])
        ax = pf.clear_axis(ax)
        ax.imshow(orig_imgs[batch_idx], cmap='gray')
        ax.set_title(r"$S_i$", fontsize = title_font_size)

        ax = plt.subplot(gs[2, 1])
        ax = pf.clear_axis(ax)
        ax.imshow(orig_recon[batch_idx], cmap='gray')
        ax.set_title(r"$\hat{S}_i$", fontsize = title_font_size)

        ax = plt.subplot(gs[1, 2])
        ax = pf.clear_axis(ax)
        ax.imshow(adv_img[batch_idx], cmap='gray')
        ax.set_title(r"$S^*$", fontsize = title_font_size)

        ax = plt.subplot(gs[2, 2])
        ax = pf.clear_axis(ax)
        ax.imshow(adv_recon[batch_idx], cmap='gray')
        ax.set_title(r"$\hat{S}^*$", fontsize = title_font_size)

        axbig = plt.subplot(gs[3:, :3])
        axbig.set_ylim([0, np.max([
          input_adv_mses[:, batch_idx],
          target_recon_mses[:, batch_idx],
          target_adv_mses[:, batch_idx],
          adv_recon_mses[:, batch_idx]])])
        axbig.tick_params('y', colors='k')
        axbig.set_xlabel("Step", fontsize=axes_font_size)
        axbig.set_ylabel("MSE", fontsize=axes_font_size)

        #Generate x idxs based on length of lines and save int
        line_x = np.arange(0, analyzer.analysis_params.adversarial_num_steps,
          analyzer.analysis_params.adversarial_save_int)

        line1 = axbig.plot(line_x, input_adv_mses[:, batch_idx], 'r', label="input to perturbed")
        line2 = axbig.plot(line_x, target_adv_mses[:, batch_idx], 'b', label="target to perturbed")
        line3 = axbig.plot(line_x, target_recon_mses[:, batch_idx], 'g', label="target to recon")
        line4 = axbig.plot(line_x, adv_recon_mses[:, batch_idx], 'k', label="perturbed to recon")

        lines = line1+line2+line3+line4
        line_labels = [l.get_label() for l in lines]

        ax = plt.subplot(gs[3, 3])
        ax = pf.clear_axis(ax)
        ax.legend(lines, line_labels, loc='upper left')

        fig.savefig(out_filename)
        plt.close("all")
