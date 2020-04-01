from DeepSparseCoding.params import param_picker as pp
from DeepSparseCoding.models.lambda_model import LambdaModel as lamb
from DeepSparseCoding.models.mlp_model import MlpModel as mlp
from DeepSparseCoding.models.mlp_lca_model import MlpLcaModel as mlp_lca
from DeepSparseCoding.models.mlp_lca_subspace_model import MlpLcaSubspaceModel as mlp_lca_subspace
from DeepSparseCoding.models.mlp_ae_model import MlpAeModel as mlp_ae
from DeepSparseCoding.models.mlp_vae_model import MlpVaeModel as mlp_vae
from DeepSparseCoding.models.mlp_sae_model import MlpSaeModel as mlp_sae
from DeepSparseCoding.models.mlp_lista_model import MlpListaModel as mlp_lista
from DeepSparseCoding.models.ica_model import IcaModel as ica
from DeepSparseCoding.models.ica_subspace_model import IcaSubspaceModel as ica_subspace
from DeepSparseCoding.models.ica_pca_model import IcaPcaModel as ica_pca
from DeepSparseCoding.models.rica_model import RicaModel as rica
from DeepSparseCoding.models.lca_model import LcaModel as lca
from DeepSparseCoding.models.lca_pca_model import LcaPcaModel as lca_pca
from DeepSparseCoding.models.lca_pca_fb_model import LcaPcaFbModel as lca_pca_fb
from DeepSparseCoding.models.lca_subspace_model import LcaSubspaceModel as lca_subspace
from DeepSparseCoding.models.lca_conv_model import LcaConvModel as lca_conv
from DeepSparseCoding.models.lista_model import ListaModel as lista
from DeepSparseCoding.models.ae_model import AeModel as ae
from DeepSparseCoding.models.dae_model import DaeModel as dae
from DeepSparseCoding.models.dae_mem_model import DaeMemModel as dae_mem
from DeepSparseCoding.models.vae_model import VaeModel as vae
from DeepSparseCoding.models.sae_model import SaeModel as sae

def get_model(model_type):
  if model_type.lower() == "lambda":
    return lamb()
  if model_type.lower() == "mlp":
    return mlp()
  if model_type.lower() == "mlp_lca":
    return mlp_lca()
  if model_type.lower() == "mlp_lca_subspace":
    return mlp_lca_subspace()
  if model_type.lower() == "mlp_ae":
    return mlp_ae()
  if model_type.lower() == "mlp_vae":
    return mlp_vae()
  if model_type.lower() == "mlp_sae":
    return mlp_sae()
  if model_type.lower() == "mlp_lista":
    return mlp_lista()
  if model_type.lower() == "ica":
    return ica()
  if model_type.lower() == "ica_subspace":
    return ica_subspace()
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
  if model_type.lower() == "lca_subspace":
    return lca_subspace()
  if model_type.lower() == "lista":
    return lista()
  if model_type.lower() == "ae":
    return ae()
  if model_type.lower() == "dae":
    return dae()
  if model_type.lower() == "dae_mem":
    return dae_mem()
  if model_type.lower() == "sae":
    return sae()
  if model_type.lower() == "vae":
    return vae()
  assert False, ("model_type did not match allowable types in models/model_picker.")

def get_model_list():
  model_list = ["lambda", "mlp", "mlp_lca", "mlp_lca_subspace", "mlp_ae", "mlp_vae", "mlp_sae",
    "mlp_lista", "ica", "ica_subspace", "rica", "lca", "lca_pca", "lca_pca_fb", "lca_conv",
    "lca_subspace", "lista", "ae", "dae", "dae_mem", "sae", "vae"]
  return model_list
