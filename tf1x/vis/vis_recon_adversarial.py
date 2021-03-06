import matplotlib
matplotlib.use('Agg')
import os
import sys
import pdb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from skimage.measure import compare_psnr

ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

from DeepSparseCoding.tf1x.data.dataset import Dataset
import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap

#List of models for analysis
analysis_list = [
  #("lca", "lca_1568_mnist"),
  ("lca", "lca_768_mnist"),
  #("vae", "vae_mnist"),
  #("vae", "deep_vae_mnist"),
  #("vae", "deep_denoising_vae_mnist"),
  #("sae", "sae_768_mnist"),
  ]

#colors for analysis_list
colors = [
  [1.0, 0.0, 0.0], #"r"
  [0.0, 0.0, 1.0], #"b"
  [0.0, 1.0, 0.0], #"g"
  [0.0, 1.0, 1.0], #"c"
  [1.0, 0.0, 1.0], #"m"
  [1.0, 1.0, 0.0], #"y"
  ]

title_font_size = 16
axes_font_size = 16

construct_recon_mult_tradeoff = True
construct_adv_examples = True
construct_over_time = True
num_output_batches = 3
recon_mult_idx = -2 #use second to last recon mult

#Base outdir for multi-network plot
outdir = "/home/slundquist/Work/Projects/vis/"
#outdir = "/home/dpaiton/Work/Projects/vis/"

class params(object):
  def __init__(self):
    self.model_type = ""
    self.model_name = ""
    self.plot_title_name = model_name.replace("_", " ").title()
    #self.save_info = "analysis_test_kurakin_targeted"
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
  return analyzer

makedir(outdir)

if construct_recon_mult_tradeoff:
  fig_unnorm, ax_unnorm = plt.subplots()
  fig_norm, ax_norm = plt.subplots()
  fig_max_change, ax_max_change = plt.subplots()

  #saved_rm = [0 for i in analysis_list]

  #TODO only do this analysis if we're sweeping
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

    analyzer = setup(analysis_params)

    num_data = analyzer.num_data
    target_img = analyzer.adversarial_target_images
    orig_img = analyzer.adversarial_images[0, 0, ...]
    orig_recon = analyzer.adversarial_recons[0, 0, ...]
    adv_img = analyzer.adversarial_images[:, -1, ...]
    adv_recon = analyzer.adversarial_recons[:, -1, ...]

    #Calculate distances
    r_mults = adv_img.shape[0]

    input_adv_vals = np.zeros((r_mults, num_data))
    target_recon_vals = np.zeros((r_mults, num_data))
    recon_adv_recon_vals = np.zeros((r_mults, num_data))
    for r_idx in range(r_mults):
      input_adv_vals[r_idx, :] = dp.cos_similarity(orig_img, adv_img[r_idx])
      target_recon_vals[r_idx, :] = dp.cos_similarity(target_img, adv_recon[r_idx])
      recon_adv_recon_vals[r_idx, :] = dp.cos_similarity(orig_recon, adv_recon[r_idx])
    input_recon_val = dp.cos_similarity(orig_img, orig_recon)


    ##These mses are in shape [num_recon_mults, num_iterations, num_batch]
    #input_adv_vals = np.array(analyzer.adversarial_input_adv_mses)[:, -1, :]
    #target_recon_vals = np.array(analyzer.adversarial_target_recon_mses)[:, -1, :]
    #input_recon_val = np.array(analyzer.adversarial_input_recon_mses)[:, 0, :]

    #reduc_dims = tuple(range(2, len(orig_recon.shape)))
    #recon_adv_recon_vals = np.mean((orig_recon - adv_recon)**2, axis=reduc_dims)

    #input_recon_val should be within threshold no matter the recon val being tested
    #Here, not identical because vae's sample latent space, so recon is non-deterministic
    #try:
    #  assert np.all(np.abs(input_recon_val[0] - np.mean(input_recon_val, axis=0)) < 1e-3)
    #except:
    #  pdb.set_trace()

    norm_target_recon_vals = target_recon_vals / input_recon_val[None, :]

    #Normalize target_recon mse by orig_recon mse
    recon_mult = np.array(analyzer.analysis_params.carlini_recon_mult)


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

    ax_max_change.errorbar(np.mean(input_adv_vals, axis=-1),
      np.mean(recon_adv_recon_vals, axis=-1),
      xerr = np.std(input_adv_vals, axis=-1),
      yerr = np.std(recon_adv_recon_vals, axis=-1),
      label = label_str, c=colors[model_idx], fmt="o")

    #ax_norm.scatter(input_adv_vals.flatten(), norm_target_recon_vals.flatten(), label=label_str, c=colors[model_idx], s=2)
    #ax_unnorm.scatter(input_adv_vals.flatten(), target_recon_vals.flatten(), label=label_str, c=colors[model_idx], s=2)

  #Draw identity line
  identity_adv_vals = np.zeros((recon_mult.shape[0], orig_img.shape[0]))
  identity_target_recon_vals = np.zeros((recon_mult.shape[0], orig_img.shape[0]))
  identity_recon_adv_recon_vals = np.zeros((recon_mult.shape[0], orig_img.shape[0]))

  #Find line with identity
  for i, rmult in enumerate(recon_mult):
    recon = rmult * orig_img + (1 - rmult) * target_img
    #identity_adv_vals[i, :] = dp.mse(recon, orig_img)
    #identity_target_recon_vals[i, :] = dp.mse(recon, target_img)
    #identity_recon_adv_recon_vals[i, :] = dp.mse(recon, orig_img)
    identity_adv_vals[i, :] = dp.cos_similarity(recon, orig_img)
    identity_target_recon_vals[i, :] = dp.cos_similarity(recon, target_img)
    identity_recon_adv_recon_vals[i, :] = dp.cos_similarity(recon, orig_img)

  #Plot on unnorm
  ax_unnorm.errorbar(np.mean(identity_adv_vals, axis=-1),
      np.mean(identity_target_recon_vals, axis=-1),
      xerr = np.std(identity_adv_vals, axis=-1),
      yerr=np.std(identity_target_recon_vals, axis=-1),
      label = "identity", c = "k", fmt = "o")

  ax_max_change.errorbar(np.mean(identity_adv_vals, axis=-1),
      np.mean(identity_recon_adv_recon_vals, axis=-1),
      xerr = np.std(identity_adv_vals, axis=-1),
      yerr=np.std(identity_recon_adv_recon_vals, axis=-1),
      label = "identity", c = "k", fmt = "o")

  ax_norm.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
  ax_norm.set_ylabel("Normalized Target Recon MSE", fontsize=axes_font_size)
  ax_norm.legend()

  ax_unnorm.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
  ax_unnorm.set_ylabel("Target Recon MSE", fontsize=axes_font_size)
  ax_unnorm.legend()

  ax_max_change.set_xlabel("Input Adv MSE", fontsize=axes_font_size)
  ax_max_change.set_ylabel("Recon AdvRecon MSE", fontsize=axes_font_size)
  ax_max_change.legend()

  # TODO: Save these in model folders (analyzer.analysis_out_dir) instead of root output directory
  fig_unnorm.savefig(outdir + "/"+analysis_params.save_info+"_recon_mult_tradeoff.png")
  fig_norm.savefig(outdir + "/"+analysis_params.save_info+"_norm_recon_mult_tradeoff.png")
  fig_max_change.savefig(outdir + "/"+analysis_params.save_info+"_max_change.png")

  plt.close('all')

if(construct_adv_examples):
  imgs = []
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()
    analyzer = setup(analysis_params)

    num_data = analyzer.num_data

    orig_img = analyzer.adversarial_images[recon_mult_idx, 0, ...].reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    orig_recon = analyzer.adversarial_recons[recon_mult_idx, 0, ...].reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    #Get adv examples from source
    adv_img = analyzer.adversarial_images[recon_mult_idx, -1, ...].reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    adv_recon = analyzer.adversarial_recons[recon_mult_idx, -1, ...].reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    targ_img = analyzer.adversarial_target_images.reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    pert = adv_img - orig_img
    imgs.append([orig_img, orig_recon, pert, adv_img, adv_recon, targ_img])

  for batch_id in range(num_output_batches):
    #Construct img table
    fig, ax = plt.subplots(len(analysis_list), 8, squeeze=False)
    plt.suptitle(analysis_params.save_info)

    ax[0, 1].set_title("orig")
    ax[0, 2].set_title("recon")
    ax[0, 3].set_title("pert")
    ax[0, 5].set_title("adv_image")
    ax[0, 6].set_title("adv_recon")
    ax[0, 7].set_title("target")

    for model_idx, (model_type, model_name) in enumerate(analysis_list):
      ax[model_idx, 0].text(1, 0.5, model_name, horizontalalignment="right", verticalalignment="center")

      ax[model_idx, 1].imshow(imgs[model_idx][0][batch_id], cmap="gray")
      ax[model_idx, 2].imshow(imgs[model_idx][1][batch_id], cmap="gray")

      pert_img = ax[model_idx, 3].imshow(imgs[model_idx][2][batch_id], cmap="gray")
      cb = fig.colorbar(pert_img, ax=ax[model_idx, 3], aspect=5)
      tick_locator = ticker.MaxNLocator(nbins=3)
      cb.locator = tick_locator
      cb.update_ticks()

      ax[model_idx, 5].imshow(imgs[model_idx][3][batch_id], cmap="gray")
      ax[model_idx, 6].imshow(imgs[model_idx][4][batch_id], cmap="gray")
      ax[model_idx, 7].imshow(imgs[model_idx][5][batch_id], cmap="gray")

      for i in range(8):
        pf.clear_axis(ax[model_idx, i])

    fig.savefig(outdir+"/adv_recon_example_"+analysis_params.save_info+"_batch_" + str(batch_id)+ ".png")
    plt.close("all")

if(construct_over_time):
  for model_idx, (model_type, model_name) in enumerate(analysis_list):
    analysis_params = params()
    analysis_params.model_type = model_type
    analysis_params.model_name = model_name
    analysis_params.plot_title_name = analysis_params.model_name.replace("_", " ").title()

    analyzer = setup(analysis_params)

    num_data = analyzer.num_data
    orig_imgs = analyzer.recon_adversarial_input_images.reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))
    target_imgs = analyzer.adversarial_target_images.reshape(
      int(num_data),
      int(np.sqrt(analyzer.model.params.num_pixels)),
      int(np.sqrt(analyzer.model.params.num_pixels)))

    for batch_idx in range(num_output_batches):
      #TODO put batch idx in filename
      out_dir = analyzer.analysis_out_dir+"/vis/adv_input_imges/"
      makedir(out_dir)
      pf.plot_image(orig_imgs[batch_idx], title="Input Image",
        save_filename=out_dir+analysis_params.save_info+"_adversarial_input_batch_"+str(batch_idx)+".png")
      out_dir = analyzer.analysis_out_dir+"/vis/adv_target_imges/"
      makedir(out_dir)
      pf.plot_image(target_imgs[batch_idx], title="Target Image",
        save_filename=out_dir+analysis_params.save_info+"_adversarial_target_batch_"+str(batch_idx)+".png")

    plot_int = 100
    if "kurakin" in analyzer.analysis_params.adversarial_attack_method:
      rm_list = enumerate([1.0]) # no recon multiplier for Kurakin attacks
    else:
      recon_mult = analyzer.analysis_params.carlini_recon_mult
      rm_list = enumerate(recon_mult)

    for i_rm, rm in rm_list:
      rm_str = "%.7f"%rm
      orig_recon = np.array(analyzer.adversarial_recons)[i_rm, 0, ...].reshape(
        int(num_data),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))

      adv_recon = np.array(analyzer.adversarial_recons)[i_rm, -1, ...].reshape(
        int(num_data),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))

      adv_img = np.array(analyzer.adversarial_images)[i_rm, -1, ...].reshape(
        int(num_data),
        int(np.sqrt(analyzer.model.params.num_pixels)),
        int(np.sqrt(analyzer.model.params.num_pixels)))

      input_adv_mses = np.array(analyzer.adversarial_input_adv_mses)[i_rm]
      target_recon_mses = np.array(analyzer.adversarial_target_recon_mses)[i_rm]
      target_adv_mses = np.array(analyzer.adversarial_target_adv_mses)[i_rm]
      adv_recon_mses = np.array(analyzer.adversarial_adv_recon_mses)[i_rm]

      for batch_idx in range(num_output_batches):
        out_dir = analyzer.analysis_out_dir+"/vis/adv_losses/"
        makedir(out_dir)
        out_filename = out_dir+\
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
        line_x = analyzer.steps_idx

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
