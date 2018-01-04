import params.param_picker as pp
from models.mlp import MLP as mlp
from models.ica import ICA as ica
from models.ica_pca import ICA_PCA as ica_pca
from models.lca import LCA as lca
from models.lca_pca import LCA_PCA as lca_pca
from models.lca_pca_fb import LCA_PCA_FB as lca_pca_fb
from models.conv_lca import CONV_LCA as conv_lca
from models.density_learner import density_learner as dl

def get_model(model_type):
  if model_type.lower() == "mlp":
    return mlp()
  if model_type.lower() == "ica":
    return ica()
  if model_type.lower() == "ica_pca":
    return ica_pca()
  if model_type.lower() == "lca":
    return lca()
  if model_type.lower() == "lca_pca":
    return lca_pca()
  if model_type.lower() == "lca_pca_fb":
    return lca_pca_fb()
  if model_type.lower() == "conv_lca":
    return conv_lca()
  if model_type.lower() == "density_learner":
    return dl()

def get_model_list():
  model_list = ["mlp", "ica", "ica_pca", "lca", "lca_pca", "lca_pca_fb", "conv_lca",
    "density_learner"]
  return model_list
