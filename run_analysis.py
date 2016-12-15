import matplotlib
matplotlib.use("Agg")

import os
#from analysis.base_analysis import Analyzer as base_analysis
#from analysis.karklin_lewicki import karklin_lewicki as kl_analysis
import utils.log_parser as log_parser

analysis_params = {
  "versions": ["0.0"],
  "model_name": "karklin_lewicki",
  "batch_index": -1,
  "eval_train": True,
  "eval_test": True,
  "eval_val": True,
  "file_ext": ".pdf",
  "device": "/cpu:0",
  # K&L specific params
  "eval_inference": True,
  "eval_density_weights": True}

log_file_loc = os.path.expanduser("~")+"/Work/Projects/test/logfiles/test_v0.0.log"

log_text = log_parser.load_file(log_file_loc)

model_params = log_parser.read_params(log_text)

model_schedule = log_parser.read_schedule(log_text)

import IPython; IPython.embed()

