import params.mlp_params as mlp
import params.ica_params as ica
import params.ica_pca_params as ica_pca
import params.rica_params as rica
import params.lca_params as lca
import params.lca_pca_params as lca_pca
import params.lca_pca_fb_params as lca_pca_fb
import params.subspace_lca_params as subspace_lca
import params.conv_lca_params as conv_lca
import params.sigmoid_autoencoder_params as sa
import params.gdn_autoencoder_params as ga
import params.conv_gdn_autoencoder_params as cga
import params.conv_gdn_decoder_params as cgd
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
    params.schedule = mlp.schedule
    return params
  if model_type.lower() == "ica":
    params = ica.params()
    params.schedule = ica.schedule
    return params
  if model_type.lower() == "ica_pca":
    params = ica_pca.params()
    params.schedule = ica_pca.schedule
    return params
  if model_type.lower() == "rica":
    params = rica.params()
    params.schedule = rica.schedule
    return params
  if model_type.lower() == "lca":
    params = lca.params()
    params.schedule = lca.schedule
    return params
  if model_type.lower() == "lca_pca":
    params = lca_pca.params()
    params.schedule = lca_pca.schedule
    return params
  if model_type.lower() == "lca_pca_fb":
    params = lca_pca_fb.params()
    params.schedule = lca_pca_fb.schedule
    return params
  if model_type.lower() == "subspace_lca":
    params = subspace_lca.params()
    params.schedule = subspace_lca.schedule
    return params
  if model_type.lower() == "conv_lca":
    params = conv_lca.params()
    params.schedule = conv_lca.schedule
    return params
  if model_type.lower() == "sigmoid_autoencoder":
    params = sa.params()
    params.schedule = sa.schedule
    return params
  if model_type.lower() == "vae":
    params = vae.params()
    params.schedule = vae.schedule
    return params
  if model_type.lower() == "gdn_autoencoder":
    params = ga.params()
    params.schedule = ga.schedule
    return params
  if model_type.lower() == "conv_gdn_autoencoder":
    params = cga.params()
    params.schedule = cga.schedule
    return params
  if model_type.lower() == "conv_gdn_decoder":
    params = cgd.params()
    params.schedule = cgd.schedule
    return params
  if model_type.lower() == "relu_autoencoder":
    params = ra.params()
    params.schedule = ra.schedule
    return params
  return False
