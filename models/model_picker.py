from models.mlp import MLP as mlp
from models.lca import LCA as lca
from models.ica import ICA as ica
#from models.karklin_lewicki import karklin_lewicki
from models.deep_sparse_coding import deep_sparse_coding as dsc

def get_model(params, schedule):
  if params["model_type"].lower() == "mlp":
    return mlp(params, schedule)
  if params["model_type"].lower() == "lca":
    return lca(params, schedule)
  if params["model_type"].lower() == "ica":
    return ica(params, schedule)
  if params["model_type"].lower() == "deep_sparse_coding":
    return dsc(params, schedule)
  if params["model_type"].lower == "karklin_lewicki":
    return karklin_lewicki(params, schedule)

def list_models():
  model_list = ["mlp", "deep_sparse_coding", "lca", "ica"]
  return model_list
