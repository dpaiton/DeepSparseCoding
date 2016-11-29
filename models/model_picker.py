from models.mlp import MLP as mlp
from models.karklin_lewicki import karklin_lewicki

def get_model(params, schedule):
  if params["model_type"] == "mlp":
    return mlp(params, schedule)
  if params["model_type"] == "karklin_lewicki":
    return karklin_lewicki(params, schedule)

def list_models():
  model_list = ["mlp", "karklin_lewicki"]
  return model_list
