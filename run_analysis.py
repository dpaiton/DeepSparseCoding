import matplotlib
matplotlib.use("Agg")

import os
import models.model_picker as mp
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
model_stats = log_parser.read_stats(log_text)

np_rand_state = np.random.RandomState(model_params["rand_seed"])

data = load_MNIST(model_params["data_dir"],
  normalize_imgs=model_params["norm_images"],
  rand_state=np_rand_state)

model = mp.get_model(model_params, model_schedule)

#_, u_t, v_t, = sess.run([model.do_inference,
#  model.u_t, model.v_t,],
#  feed_dict)
#print(np.max(u_t))
#print("\n")
#print(np.max(v_t))
#print("\n")

import IPython; IPython.embed(); raise SystemExit

