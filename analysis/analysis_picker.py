from analysis.mlp_analyzer import MlpAnalyzer as mlp_analyzer
from analysis.ica_analyzer import IcaAnalyzer as ica_analyzer
from analysis.lca_analyzer import LcaAnalyzer as lca_analyzer
from analysis.lca_subspace_analyzer import LcaSubspaceAnalyzer as sub_lca_analyzer
from analysis.lca_pca_analyzer import LcaPcaAnalyzer as lca_pca_analyzer
from analysis.lca_conv_analyzer import LcaConvAnalyzer as conv_lca_analyzer
from analysis.lista_analyzer import ListaAnalyzer as lista_analyzer
from analysis.rica_analyzer import RicaAnalyzer as rica_analyzer
from analysis.ae_analyzer import AeAnalyzer as ae_analyzer
from analysis.sae_analyzer import SaeAnalyzer as sae_analyzer
from analysis.vae_analyzer import VaeAnalyzer as vae_analyzer
from analysis.gdn_autoencoder_analyzer import GaAnalyzer as ga_analyzer

def get_analyzer(model_type):
  if model_type.lower() in ["mlp", "mlp_ae", "mlp_vae", "mlp_lca", "mlp_sae", "mlp_lista"]:
    return mlp_analyzer()
  if model_type.lower() == "ica":
    return ica_analyzer()
  if model_type.lower() == "rica":
    return rica_analyzer()
  if model_type.lower() == "lca":
    return lca_analyzer()
  if model_type.lower() == "lca_subspace":
    return sub_lca_analyzer()
  if model_type.lower() == "lca_pca":
    return lca_pca_analyzer()
  if model_type.lower() == "lca_conv":
    return conv_lca_analyzer()
  if model_type.lower() == "lista":
    return lista_analyzer()
  if model_type.lower() == "ae":
    return ae_analyzer()
  if model_type.lower() == "sae":
    return sae_analyzer()
  if model_type.lower() == "gdn_autoencoder":
    return ga_analyzer()
  if model_type.lower() == "vae":
    return vae_analyzer()
  assert False, ("model_type did not match allowable types in analysis/analysis_picker.")
