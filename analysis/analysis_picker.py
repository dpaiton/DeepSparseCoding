def get_analyzer(params):
  if params["model_type"].lower() == "mlp":
    from analysis.mlp_analyzer import mlp as analyzer
  if params["model_type"].lower() == "ica":
    from analysis.ica_analyzer import ica as analyzer
  if params["model_type"].lower() == "lca":
    from analysis.lca_analyzer import lca as analyzer
  if params["model_type"].lower() == "conv_lca":
    from analysis.conv_lca_analyzer import conv_lca as analyzer
  if params["model_type"].lower() == "dsc":
    from analysis.dsc_analyzer import dsc as analyzer
  if params["model_type"].lower() == "density_learner":
    from analysis.density_analyzer import density_analyzer as analyzer
  return analyzer(params)
