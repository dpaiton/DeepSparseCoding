from params.param_picker import get_params

def get_model(model_type):
  params, schedule = get_params(model_type)
  if model_type.lower() == "mlp":
    from models.mlp import MLP as mlp
    return (mlp(params, schedule), params, schedule)
  if model_type.lower() == "lca":
    from models.lca import LCA as lca
    return (lca(params, schedule), params, schedule)
  if model_type.lower() == "ica":
    from models.ica import ICA as ica
    return (ica(params, schedule), params, schedule)
  if model_type.lower() == "deep_sparse_coding":
    from models.deep_sparse_coding import deep_sparse_coding as dsc
    return (dsc(params, schedule), params, schedule)

def list_models():
  model_list = ["mlp", "deep_sparse_coding", "lca", "ica"]
  return model_list
