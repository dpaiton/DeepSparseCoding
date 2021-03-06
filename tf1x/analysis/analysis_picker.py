from DeepSparseCoding.tf1x.analysis.lambda_analyzer import LambdaAnalyzer as lamb
from DeepSparseCoding.tf1x.analysis.mlp_analyzer import MlpAnalyzer as mlp
from DeepSparseCoding.tf1x.analysis.ica_analyzer import IcaAnalyzer as ica
from DeepSparseCoding.tf1x.analysis.lca_analyzer import LcaAnalyzer as lca
from DeepSparseCoding.tf1x.analysis.lca_subspace_analyzer import LcaSubspaceAnalyzer as sub_lca
from DeepSparseCoding.tf1x.analysis.lca_pca_analyzer import LcaPcaAnalyzer as lca_pca
from DeepSparseCoding.tf1x.analysis.lca_conv_analyzer import LcaConvAnalyzer as conv_lca
from DeepSparseCoding.tf1x.analysis.lista_analyzer import ListaAnalyzer as lista
from DeepSparseCoding.tf1x.analysis.rica_analyzer import RicaAnalyzer as rica
from DeepSparseCoding.tf1x.analysis.ae_analyzer import AeAnalyzer as ae
from DeepSparseCoding.tf1x.analysis.sae_analyzer import SaeAnalyzer as sae
from DeepSparseCoding.tf1x.analysis.vae_analyzer import VaeAnalyzer as vae
from DeepSparseCoding.tf1x.analysis.gdn_autoencoder_analyzer import GaAnalyzer as ga

def get_analyzer(model_type):
  if model_type.lower() in ["mlp", "mlp_ae", "mlp_vae", "mlp_lca", "mlp_sae", "mlp_lista"]:
    return mlp()
  if model_type.lower() == "lambda":
    return lamb()
  if model_type.lower() in ["ica", "ica_subspace"]:
    return ica()
  if model_type.lower() == "rica":
    return rica()
  if model_type.lower() == "lca":
    return lca()
  if model_type.lower() == "lca_subspace":
    return sub_lca()
  if model_type.lower() == "lca_pca":
    return lca_pca()
  if model_type.lower() == "lca_conv":
    return conv_lca()
  if model_type.lower() == "lista":
    return lista()
  if model_type.lower() == "ae":
    return ae()
  if model_type.lower() == "sae":
    return sae()
  if model_type.lower() == "gdn_autoencoder":
    return ga()
  if model_type.lower() == "vae":
    return vae()
  assert False, ("model_type did not match allowable types in analysis/analysis_picker.")
