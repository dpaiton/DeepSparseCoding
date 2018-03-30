import params.mlp_params as mlp_params
import params.ica_params as ica_params
import params.ica_pca_params as ica_pca_params
import params.lca_params as lca_params
import params.lca_pca_params as lca_pca_params
import params.lca_pca_fb_params as lca_pca_fb_params
import params.conv_lca_params as conv_lca_params
import params.dsc_params as dsc_params
import params.density_learner_params as density_params
import params.entropy_sc_params as esc_params
import params.gradient_sc_params as gsc_params

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
    return mlp_params.params, mlp_params.schedule
  if model_type.lower() == "ica":
    return ica_params.params, ica_params.schedule
  if model_type.lower() == "ica_pca":
    return ica_pca_params.params, ica_pca_params.schedule
  if model_type.lower() == "lca":
    return lca_params.params, lca_params.schedule
  if model_type.lower() == "lca_pca":
    return lca_pca_params.params, lca_pca_params.schedule
  if model_type.lower() == "lca_pca_fb":
    return lca_pca_fb_params.params, lca_pca_fb_params.schedule
  if model_type.lower() == "conv_lca":
    return conv_lca_params.params, conv_lca_params.schedule
  if model_type.lower() == "dsc":
    return dsc_params.params, dsc_params.schedule
  if model_type.lower() == "density_learner":
    return density_params.params, density_params.schedule
  if model_type.lower() == "entropy_sc":
    return esc_params.params, esc_params.schedule
  if model_type.lower() == "gradient_sc":
    return gsc_params.params, gsc_params.schedule
  return False
