from analysis.lca_analyzer import LCA as lca_analyzer
from analysis.lca_pca_analyzer import LCA_PCA as lca_pca_analyzer
from analysis.dsc_analyzer import DSC as dsc_analyzer
from analysidscs.density_analyzer import density_analyzer as density_analyzer

def get_analyzer(params):
  if params["model_type"].lower() == "lca":
    return lca_analyzer(params)
  if params["model_type"].lower() == "lca_pca":
    return lca_pca_analyzer(params)
  if params["model_type"].lower() == "DSC":
    return dsc_analyzer(params)
  if params["model_type"].lower() == "density_learner":
    return density_analyzer(params)
