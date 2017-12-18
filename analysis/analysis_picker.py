from analysis.lca_analyzer import LCA as lca_analyzer
from analysis.lca_pca_analyzer import LCA_PCA as lca_pca_analyzer
from analysis.lca_pca_fb_analyzer import LCA_PCA_FB as lca_pca_fb_analyzer
from analysis.conv_lca_analyzer import CONV_LCA as conv_lca_analyzer
from analysis.dsc_analyzer import DSC as dsc_analyzer
from analysis.density_analyzer import density_analyzer as density_analyzer

def get_analyzer(params):
  if params["model_type"].lower() == "lca":
    return lca_analyzer(params)
  if params["model_type"].lower() == "lca_pca":
    return lca_pca_analyzer(params)
  if params["model_type"].lower() == "lca_pca_fb":
    return lca_pca_fb_analyzer(params)
  if params["model_type"].lower() == "conv_lca":
    return conv_lca_analyzer(params)
  if params["model_type"].lower() == "DSC":
    return dsc_analyzer(params)
  if params["model_type"].lower() == "density_learner":
    return density_analyzer(params)
  assert False, ("model_type did not match allowable types in analysis.get_analyzer.")
