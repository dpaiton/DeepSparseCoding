import params.mlp_params as mlp
import params.ica_params as ica
import params.ica_pca_params as ica_pca
import params.rica_params as rica
import params.lca_params as lca
import params.lca_pca_params as lca_pca
import params.lca_pca_fb_params as lca_pca_fb
import params.lca_subspace_params as lca_subspace
import params.lca_conv_params as lca_conv
import params.lista_params as lista
import params.sigmoid_autoencoder_params as sa
import params.gdn_autoencoder_params as ga
import params.gdn_conv_autoencoder_params as cga
import params.gdn_conv_decoder_params as cgd
import params.relu_autoencoder_params as ra
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
    params = mlp.params()
    return params
  if model_type.lower() == "ica":
    params = ica.params()
    return params
  if model_type.lower() == "ica_pca":
    params = ica_pca.params()
    return params
  if model_type.lower() == "rica":
    params = rica.params()
    return params
  if model_type.lower() == "lca":
    params = lca.params()
    return params
  if model_type.lower() == "lca_pca":
    params = lca_pca.params()
    return params
  if model_type.lower() == "lca_pca_fb":
    params = lca_pca_fb.params()
    return params
  if model_type.lower() == "lca_subspace":
    params = lca_subspace.params()
    return params
  if model_type.lower() == "lca_conv":
    params = lca_conv.params()
    return params
  if model_type.lower() == "lista":
    params = lista.params()
    return params
  if model_type.lower() == "sigmoid_autoencoder":
    params = sa.params()
    return params
  if model_type.lower() == "vae":
    params = vae.params()
    return params
  if model_type.lower() == "gdn_autoencoder":
    params = ga.params()
    return params
  if model_type.lower() == "gdn_conv_autoencoder":
    params = cga.params()
    return params
  if model_type.lower() == "gdn_conv_decoder":
    params = cgd.params()
    return params
  if model_type.lower() == "relu_autoencoder":
    params = ra.params()
    return params
  return False
