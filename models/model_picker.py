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
from models.lca_mlp_model import LcaMlpModel as lca_mlp
from models.lista_model import ListaModel as lista
from models.sigmoid_autoencoder_model import SigmoidAutoencoderModel as sa
from models.gdn_autoencoder_model import GdnAutoencoderModel as ga
from models.gdn_conv_autoencoder_model import GdnConvAutoencoderModel as ga_conv
from models.gdn_conv_decoder_model import GdnConvDecoderModel as gd_conv
from models.relu_autoencoder_model import ReluAutoencoderModel as ra
from models.vae_model import VaeModel as vae
from models.ae_model import AeModel as ae

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
  if model_type.lower() == "lca_conv":
    return lca_conv()
  if model_type.lower() == "lca_mlp":
    return lca_mlp()
  if model_type.lower() == "lca_subspace":
    return lca_subspace()
  if model_type.lower() == "lista":
    return lista()
  if model_type.lower() == "sigmoid_autoencoder":
    return sa()
  if model_type.lower() == "gdn_autoencoder":
    return ga()
  if model_type.lower() == "gdn_conv_autoencoder":
    return ga_conv()
  if model_type.lower() == "gdn_conv_decoder":
    return gd_conv()
  if model_type.lower() == "relu_autoencoder":
    return ra()
  if model_type.lower() == "vae":
    return vae()
  if model_type.lower() == "ae":
    return ae()

def get_model_list():
  model_list = ["mlp", "ica", "ica_pca", "rica", "lca", "lca_pca", "lca_pca_fb", "lca_conv",
    "lca_mlp", "lca_subspace", "lista", "sigmoid_autoencoder", "gdn_autoencoder",
    "gdn_conv_autoencoder", "gdn_conv_decoder", "relu_autoencoder", "ae", "vae"]
  return model_list
