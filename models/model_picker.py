import params.param_picker as pp

def get_params(model_type):
  return pp.get_params(model_type)

def get_model(model_type, params=None, schedule=None):
  if params is None:
    params, _ = get_params(model_type) 
  if schedule is None:
    _, schedule  = get_params(model_type) 
  if model_type.lower() == "mlp":
    from models.mlp import MLP as mlp
    return mlp(params, schedule)
  if model_type.lower() == "ica":
    from models.ica import ICA as ica
    return ica(params, schedule)
  if model_type.lower() == "lca":
    from models.lca import LCA as lca
    return lca(params, schedule)
  if model_type.lower() == "conv_lca":
    from models.conv_lca import conv_LCA as conv_lca
    return conv_lca(params, schedule)
  if model_type.lower() == "deep_sparse_coding":
    from models.deep_sparse_coding import deep_sparse_coding as dsc
    return dsc(params, schedule)

def get_model_list():
  model_list = ["mlp", "ica", "lca", "conv_lca", "deep_sparse_coding"]
  return model_list
