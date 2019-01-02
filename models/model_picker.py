import params.param_picker as pp
from models.mlp_model import MlpModel as mlp
from models.ica_model import IcaModel as ica
from models.ica_pca_model import IcaPcaModel as ica_pca
from models.rica_model import RicaModel as rica
from models.lca_model import LcaModel as lca
from models.lca_pca_model import LcaPcaModel as lca_pca
from models.lca_pca_fb_model import LcaPcaFbModel as lca_pca_fb
from models.lca_subspace_model import LcaSubspaceModel as lca_subspace
from models.lca_conv_model import LcaConvModel as lca_conv
from models.sigmoid_autoencoder_model import SigmoidAutoencoderModel as sa
from models.gdn_autoencoder_model import GdnAutoencoderModel as ga
from models.conv_gdn_autoencoder_model import ConvGdnAutoencoderModel as cga
from models.conv_gdn_decoder_model import ConvGdnDecoderModel as cgd
from models.relu_autoencoder_model import ReluAutoencoderModel as ra
from models.vae_model import VaeModel as vae

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
