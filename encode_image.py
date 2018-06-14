import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
from data.nat_dataset import Dataset

## Specify model type and data type
model_type = "conv_gdn_autoencoder"
data_type = "nat_images"

## Import params
params, schedule = pp.get_params(model_type)
if "rand_seed" in params.keys():
  params["rand_state"] = np.random.RandomState(params["rand_seed"])
params["data_type"] = data_type
params["data_shape"] = [params["im_size_y"], params["im_size_x"], 1]

# Specific params to encoding
params["batch_size"] = 2 # batch size of 1 is broken
params["data_file"] = "/home/dpaiton/tmp_file_loc.txt"

## Import data
data = {"train": Dataset(params["data_file"], params)}

schedule[0]["num_batches"] = 1
schedule[0]["decay_steps"] = [1 for _ in range(len(schedule[0]["weights"]))]

## Import model
model = mp.get_model(model_type)
model.setup(params, schedule)

## Write model weight savers for checkpointing and visualizing graph
model.write_saver_defs()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=model.graph) as sess:
  ## Need to provide shape if batch_size is used in graph
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros([params["batch_size"]]+params["data_shape"], dtype=np.float32)})

  sess.graph.finalize() # Graph is read-only after this statement
  model.write_graph(sess.graph_def)

  cp_load_prefix = (model.cp_load_dir+model.cp_load_name+"_v"+model.cp_load_ver
    +"_weights")
  cp_load_file = tf.train.latest_checkpoint(cp_load_prefix)
  model.load_weights(sess, "/home/dpaiton/Work/Projects/conv_gdn_autoencoder/checkpoints/conv_gdn_autoencoder_v0.0_weights-44640")

  data_batch = data["train"].next_batch(model.batch_size)
  input_data = data_batch[0]
  input_labels = data_batch[1]
  ## Get feed dictionary for placeholders
  feed_dict = model.get_feed_dict(input_data, input_labels)
  mem_std_eps = np.random.standard_normal((model.params["batch_size"],
     model.params["n_mem"])).astype(np.float32)
  feed_dict[model.memristor_std_eps] = mem_std_eps

  latent_encoding = sess.run(model.pre_mem, feed_dict)[0]

np.savez("/home/dpaiton/image_rram_encoding.npz", data=latent_encoding)

boo = np.load("/home/dpaiton/image_rram_encoding.npz")["data"]
import IPython; IPython.embed()
