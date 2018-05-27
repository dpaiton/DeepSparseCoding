import params.param_picker as pp
from models.mlp import MLP as mlp
from models.ica import ICA as ica
from models.ica_pca import ICA_PCA as ica_pca
from models.gradient_sc import Gradient_SC as gsc
from models.entropy_sc import Entropy_SC as esc
from models.lca import LCA as lca
from models.lca_pca import LCA_PCA as lca_pca
from models.lca_pca_fb import LCA_PCA_FB as lca_pca_fb
from models.conv_lca import CONV_LCA as conv_lca
from models.sigmoid_autoencoder import Sparse_Autoencoder as sa
from models.gdn_autoencoder import GDN_Autoencoder as ga
from models.relu_autoencoder import ReLU_Autoencoder as ra
from models.density_learner import Density_Learner as dl

def get_model(model_type):
  if model_type.lower() == "mlp":
    return mlp()
  if model_type.lower() == "ica":
    return ica()
  if model_type.lower() == "ica_pca":
    return ica_pca()
  if model_type.lower() == "gradient_sc":
    return gsc()
  if model_type.lower() == "entropy_sc":
    return esc()
  if model_type.lower() == "lca":
    return lca()
  if model_type.lower() == "lca_pca":
    return lca_pca()
  if model_type.lower() == "lca_pca_fb":
    return lca_pca_fb()
  if model_type.lower() == "conv_lca":
    return conv_lca()
  if model_type.lower() == "sigmoid_autoencoder":
    return sa()
  if model_type.lower() == "gdn_autoencoder":
    return ga()
  if model_type.lower() == "relu_autoencoder":
    return ra()
  if model_type.lower() == "density_learner":
    return dl()

def get_model_list():
  model_list = ["mlp", "ica", "ica_pca", "gradient_sc", "entropy_sc", "lca", "lca_pca",
    "lca_pca_fb", "conv_lca", "sigmoid_autoencoder", "gdn_autoencoder", "relu_autoencoder",
    "density_learner"]
  return model_list
