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
    from params.mlp_params import params, schedule
  if model_type.lower() == "lca":
    from params.lca_params import params, schedule
  if model_type.lower() == "ica":
    from params.ica_params import params, schedule
  if model_type.lower() == "dsc":
    from params.dsc_params import params, schedule
  return params, schedule
