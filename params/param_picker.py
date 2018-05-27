import params.mlp_params as mlp
import params.ica_params as ica
import params.ica_pca_params as ica_pca
import params.gradient_sc_params as gsc
import params.entropy_sc_params as esc
import params.lca_params as lca
import params.lca_pca_params as lca_pca
import params.lca_pca_fb_params as lca_pca_fb
import params.conv_lca_params as conv_lca
import params.sparse_autoencoder_params as sa
import params.gdn_autoencoder_params as ga
import params.relu_autoencoder_params as ra
import params.density_learner_params as density

"""
Get function that returns the corresponding parameter and schedule files
Inputs:
  model_type: [str] containing the type of model to load.
    Current allowed values are: mlp, lca, ica, dsc
Outputs:
  params: [dict] containing params defined in the corresponding file
  schedule: [list] of [dict] containing the learning schedule from the same file
"""
def get_params(model_type):
  if model_type.lower() == "mlp":
    return mlp.params, mlp.schedule
  if model_type.lower() == "ica":
    return ica.params, ica.schedule
  if model_type.lower() == "ica_pca":
    return ica_pca.params, ica_pca.schedule
  if model_type.lower() == "gradient_sc":
    return gsc.params, gsc.schedule
  if model_type.lower() == "entropy_sc":
    return esc.params, esc.schedule
  if model_type.lower() == "lca":
    return lca.params, lca.schedule
  if model_type.lower() == "lca_pca":
    return lca_pca.params, lca_pca.schedule
  if model_type.lower() == "lca_pca_fb":
    return lca_pca_fb.params, lca_pca_fb.schedule
  if model_type.lower() == "conv_lca":
    return conv_lca.params, conv_lca.schedule
  if model_type.lower() == "density_learner":
    return density.params, density.schedule
  if model_type.lower() == "sparse_autoencoder":
    return sa.params, sa.schedule
  if model_type.lower() == "gdn_autoencoder":
    return ga.params, ga.schedule
  if model_type.lower() == "relu_autoencoder":
    return ra.params, ra.schedule
  return False
