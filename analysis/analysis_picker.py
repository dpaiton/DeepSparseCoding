from analysis.ica_analyzer import ICA_Analyzer as ica_analyzer
from analysis.lca_analyzer import LCA_Analyzer as lca_analyzer
from analysis.lca_pca_analyzer import LCA_PCA_Analyzer as lca_pca_analyzer
from analysis.lca_pca_fb_analyzer import LCA_PCA_FB_Analyzer as lca_pca_fb_analyzer
from analysis.conv_lca_analyzer import CONV_LCA_Analyzer as conv_lca_analyzer

def get_analyzer(params):
  if params["model_type"].lower() == "ica":
    return ica_analyzer(params)
  if params["model_type"].lower() == "lca":
    return lca_analyzer(params)
  if params["model_type"].lower() == "lca_pca":
    return lca_pca_analyzer(params)
  if params["model_type"].lower() == "lca_pca_fb":
    return lca_pca_fb_analyzer(params)
  if params["model_type"].lower() == "conv_lca":
    return conv_lca_analyzer(params)
  assert False, ("model_type did not match allowable types in analysis.get_analyzer.")
