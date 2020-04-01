from DeepSparseCoding.tf1x.params import base_params as base
from DeepSparseCoding.tf1x.params import lambda_params as lamb
from DeepSparseCoding.tf1x.params import mlp_params as mlp
from DeepSparseCoding.tf1x.params import mlp_lca_subspace_params as mlp_lca_subspace
from DeepSparseCoding.tf1x.params import mlp_lca_params as mlp_lca
from DeepSparseCoding.tf1x.params import mlp_ae_params as mlp_ae
from DeepSparseCoding.tf1x.params import mlp_vae_params as mlp_vae
from DeepSparseCoding.tf1x.params import mlp_sae_params as mlp_sae
from DeepSparseCoding.tf1x.params import mlp_lista_params as mlp_lista
from DeepSparseCoding.tf1x.params import ica_params as ica
from DeepSparseCoding.tf1x.params import ica_pca_params as ica_pca
from DeepSparseCoding.tf1x.params import ica_subspace_params as ica_subspace
from DeepSparseCoding.tf1x.params import rica_params as rica
from DeepSparseCoding.tf1x.params import lca_params as lca
from DeepSparseCoding.tf1x.params import lca_pca_params as lca_pca
from DeepSparseCoding.tf1x.params import lca_pca_fb_params as lca_pca_fb
from DeepSparseCoding.tf1x.params import lca_subspace_params as lca_subspace
from DeepSparseCoding.tf1x.params import lca_conv_params as lca_conv
from DeepSparseCoding.tf1x.params import lista_params as lista
from DeepSparseCoding.tf1x.params import ae_params as ae
from DeepSparseCoding.tf1x.params import dae_params as dae
from DeepSparseCoding.tf1x.params import dae_mem_params as dae_mem
from DeepSparseCoding.tf1x.params import sae_params as sae
from DeepSparseCoding.tf1x.params import vae_params as vae

"""
Get function that returns the corresponding parameter and schedule files
Inputs:
  model_type: [str] containing the type of model to load.
Outputs:
  params: [dict] containing params defined in the corresponding file
  schedule: [list] of [dict] containing the learning schedule from the same file
"""
def get_params(model_type):
  if model_type.lower() == "lambda":
    return lamb.params()
  if model_type.lower() == "mlp":
    return mlp.params()
  if model_type.lower() == "mlp_lca_subspace":
    return mlp_lca_subspace.params()
  if model_type.lower() == "mlp_lca":
    return mlp_lca.params()
  if model_type.lower() == "mlp_ae":
    return mlp_ae.params()
  if model_type.lower() == "mlp_vae":
    return mlp_vae.params()
  if model_type.lower() == "mlp_sae":
    return mlp_sae.params()
  if model_type.lower() == 'mlp_lista':
    return mlp_lista.params()
  if model_type.lower() == "ica":
    return ica.params()
  if model_type.lower() == "ica_pca":
    return ica_pca.params()
  if model_type.lower() == "ica_subspace":
    return ica_subspace.params()
  if model_type.lower() == "rica":
    return rica.params()
  if model_type.lower() == "lca":
    return lca.params()
  if model_type.lower() == "lca_pca":
    return lca_pca.params()
  if model_type.lower() == "lca_pca_fb":
    return lca_pca_fb.params()
  if model_type.lower() == "lca_subspace":
    return lca_subspace.params()
  if model_type.lower() == "lca_conv":
    return lca_conv.params()
  if model_type.lower() == "lista":
    return lista.params()
  if model_type.lower() == "ae":
    return ae.params()
  if model_type.lower() == "dae":
    return dae.params()
  if model_type.lower() == "dae_mem":
    return dae_mem.params()
  if model_type.lower() == "sae":
    return sae.params()
  if model_type.lower() == "vae":
    return vae.params()
  assert False, (model_type+" is not a supported model_type")

def list_all_params():
  all_params = [
    base,
    lamb,
    mlp,
    mlp_lca_subspace,
    mlp_lca,
    mlp_ae,
    mlp_vae,
    mlp_sae,
    mlp_lista,
    ica,
    ica_pca,
    ica_subspace,
    rica,
    lca,
    lca_pca,
    lca_pca_fb,
    lca_subspace,
    lca_conv,
    lista,
    ae,
    dae,
    dae_mem,
    sae,
    vae]
  param_names = []
  for params_obj in all_params:
    params_dict = params_obj.params().__dict__
    for param_name in list(params_dict.keys()):
      param_names.append(param_name)
  return sorted(set(param_names))

if __name__ == "__main__":
  print("\n".join(list_all_params()))
