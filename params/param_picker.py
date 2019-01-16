import params.mlp_params as mlp
import params.mlp_lca_params as mlp_lca
import params.mlp_vae_params as mlp_vae
#import params.mlp_sae_params as mlp_sae
import params.ica_params as ica
import params.ica_pca_params as ica_pca
import params.rica_params as rica
import params.lca_params as lca
import params.lca_pca_params as lca_pca
import params.lca_pca_fb_params as lca_pca_fb
import params.lca_subspace_params as lca_subspace
import params.lca_conv_params as lca_conv
import params.lista_params as lista
import params.fflista_params as fflista
import params.gdn_autoencoder_params as ga
import params.gdn_conv_autoencoder_params as cga
import params.gdn_conv_decoder_params as cgd
import params.relu_autoencoder_params as ra
import params.ae_params as ae
import params.sae_params as sae
import params.vae_params as vae

"""
Get function that returns the corresponding parameter and schedule files
Inputs:
  model_type: [str] containing the type of model to load.
Outputs:
  params: [dict] containing params defined in the corresponding file
  schedule: [list] of [dict] containing the learning schedule from the same file
"""
def get_params(model_type):
  if model_type.lower() == "mlp":
    return mlp.params()
  if model_type.lower() == "mlp_lca":
    return mlp_lca.params()
  if model_type.lower() == "mlp_vae":
    return mlp_vae.params()
  #if model_type.lower() == "mlp_sae":
  #  return mlp_sae.params()
  if model_type.lower() == "ica":
    return ica.params()
  if model_type.lower() == "ica_pca":
    return ica_pca.params()
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
  if model_type.lower() == "fflista":
    return fflista.params()
  if model_type.lower() == "ae":
    return ae.params()
  if model_type.lower() == "sae":
    return sae.params()
  if model_type.lower() == "vae":
    return vae.params()
  if model_type.lower() == "gdn_autoencoder":
    return ga.params()
  if model_type.lower() == "gdn_conv_autoencoder":
    return cga.params()
  if model_type.lower() == "gdn_conv_decoder":
    return cgd.params()
  if model_type.lower() == "relu_autoencoder":
    return ra.params()
  assert False, (model_type+" is not a supported model_type")
