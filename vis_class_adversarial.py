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

class params(object):
  #model_type = "sigmoid_autoencoder"
  #model_name = "sigmoid_autoencoder"
  model_type = "mlp"
  model_name = "mlp_mnist"
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
  makedir(analyzer.analysis_out_dir+"/vis/adversarial_recons/")
  makedir(analyzer.analysis_out_dir+"/vis/adversarial_stims/")
  return analyzer


analysis_params = params()
analyzer = setup(analysis_params)

orig_img = analyzer.adversarial_input_image.reshape(int(np.sqrt(analyzer.model.num_pixels)),
  int(np.sqrt(analyzer.model.num_pixels)))
pf.plot_image(orig_img, title="Input Image",
  save_filename=analyzer.analysis_out_dir+"/vis/adversarial_input.png")

plot_int = 100

for step, (stim, output) in enumerate(zip(analyzer.adversarial_images, analyzer.adversarial_outputs)):
  if(step % plot_int == 0):
    adv_img = stim.reshape(int(np.sqrt(analyzer.model.num_pixels)),int(np.sqrt(analyzer.model.num_pixels)))
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(adv_img, cmap='gray')
    axarr[0] = pf.clear_axis(axarr[0])
    axarr[1].bar(np.arange(analyzer.model.params.num_classes), output[0])
    axarr[1].set_ylim([0, 1])
    f.savefig(analyzer.analysis_out_dir+"/vis/adversarial_stims/stim_step_"+str(step)+".png")

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
