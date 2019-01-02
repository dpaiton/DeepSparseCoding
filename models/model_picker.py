import params.param_picker as pp
from models.mlp import MLP as mlp
from models.ica import ICA as ica
from models.ica_pca import ICA_PCA as ica_pca
from models.rica import RICA as rica
from models.lca import LCA as lca
from models.lca_pca import LCA_PCA as lca_pca
from models.lca_pca_fb import LCA_PCA_FB as lca_pca_fb
from models.lca_subspace import LCA_SUBSPACE as lca_subspace
from models.lca_conv import LCA_CONV as lca_conv
from models.sigmoid_autoencoder import Sigmoid_Autoencoder as sa
from models.gdn_autoencoder import GDN_Autoencoder as ga
from models.conv_gdn_autoencoder import Conv_GDN_Autoencoder as cga
from models.conv_gdn_decoder import Conv_GDN_Decoder as cgd
from models.relu_autoencoder import ReLU_Autoencoder as ra
from models.vae import VAE as vae

def get_model(model_type):
  if model_type.lower() == "mlp":
    return mlp()
  if model_type.lower() == "ica":
    return ica()
  if model_type.lower() == "ica_pca":
    return ica_pca()
  if model_type.lower() == "rica":
    return rica()
  if model_type.lower() == "lca":
    return lca()
  if model_type.lower() == "lca_pca":
    return lca_pca()
  if model_type.lower() == "lca_pca_fb":
    return lca_pca_fb()
  if model_type.lower() == "lca_subspace":
    return lca_subspace()
  if model_type.lower() == "lca_conv":
    return lca_conv()
  if model_type.lower() == "sigmoid_autoencoder":
    return sa()
  if model_type.lower() == "gdn_autoencoder":
    return ga()
  if model_type.lower() == "conv_gdn_autoencoder":
    return cga()
  if model_type.lower() == "conv_gdn_decoder":
    return cgd()
  if model_type.lower() == "relu_autoencoder":
    return ra()
  if model_type.lower() == "vae":
    return vae()

def get_model_list():
  model_list = ["mlp", "ica", "ica_pca", "rica", "lca", "lca_pca",
    "lca_pca_fb", "lca_conv", "sigmoid_autoencoder", "gdn_autoencoder", "conv_gdn_autoencoder",
    "conv_gdn_decoder", "relu_autoencoder", "vae"]
  return model_list
