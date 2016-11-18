from models.mlp import MLP as mlp

def get_model(params, schedule):
  if params["model_type"] == "mlp":
    return mlp(params, schedule)

def list_models():
  model_list = ["mlp"]
  return model_list
