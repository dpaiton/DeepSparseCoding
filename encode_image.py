import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
from utils.logger import Logger
from data.nat_dataset import Dataset

## Specify model type and data type
model_type = "conv_gdn_autoencoder"
model_name = "conv_gdn_autoencoder_dropout"
model_version = "1.0"
data_type = "nat_images"

## Import params
model_log_file = "/home/dpaiton/Work/Projects/"+model_name+"/logfiles/"+model_name+"_v"+model_version+".log"
model_logger = Logger(model_log_file, overwrite=False)
model_log_text = model_logger.load_file()
model_params = model_logger.read_params(model_log_text)[-1]
num_logged_input_channels = len(model_params.input_channels)
model_params.input_channels = model_params.input_channels[:num_logged_input_channels//2]
num_logged_output_channels = len(model_params.output_channels)
model_params.output_channels = model_params.output_channels[:num_logged_output_channels//2]
num_logged_strides = len(model_params.strides)
model_params.strides = model_params.strides[:num_logged_strides//2]
model_schedule = model_logger.read_schedule(model_log_text)
if hasattr(params, "rand_seed"):
  model_params.rand_state= np.random.RandomState(model_params.rand_seed)
model_params.data_type = data_type
model_params.data_shape = [model_params.im_size_y, model_params.im_size_x, 1]
model_params.gdn_w_init_const = 0.1
model_params.gdn_b_init_const = 0.1
model_params.gdn_w_thresh_min = 1e-3
model_params.gdn_b_thresh_min = 1e-3
model_params.gdn_eps = 1e-6

# Specific params to encoding
model_params.batch_size = 3 # batch size of 1 is broken
model_params.data_file = "/home/dpaiton/IEDM/tmp_encoding_imgs.txt"

## Import data
data = {"train": Dataset(model_params)}

model_schedule[0]["num_batches"] = 1
model_schedule[0]["decay_steps"] = [1 for _ in range(len(model_schedule[0]["weights"]))]

## Import model
model = mp.get_model(model_type)
model.setup(model_params)

## Write model weight savers for checkpointing and visualizing graph
model.write_saver_defs()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=config, graph=model.graph) as sess:
  ## Need to provide shape if batch_size is used in graph
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros([model.batch_size]+model.data_shape, dtype=np.float32)})

  sess.graph.finalize() # Graph is read-only after this statement
  model.write_graph(sess.graph_def)

  #model.load_weights(sess,
  #  "/home/dpaiton/Work/Projects/"+model_name+"/checkpoints/"+model_name+"_v"+model.version+"_weights-100000")
  model.load_weights(sess,
    "/home/dpaiton/Work/Projects/"+model_name+"/checkpoints/"+model_name+"_v"+model.version+"_weights-892800")

  data_batch = data["train"].next_batch(model.batch_size)
  input_data = data_batch[0]
  input_labels = data_batch[1]
  ## Get feed dictionary for placeholders
  feed_dict = model.get_feed_dict(input_data, input_labels)
  mem_std_eps = np.random.standard_normal((model.batch_size,
     model.n_mem)).astype(np.float32)
  feed_dict[model.memristor_std_eps] = mem_std_eps

  latent_encodings = sess.run(model.a_sig, feed_dict)
  recon = sess.run(model.u_list[-1], feed_dict)


for img_id in range(len(latent_encodings)):
  np.savez("/home/dpaiton/IEDM/image_recon-"+str(img_id)+".npz", data=recon[img_id, ...])
  np.savez("/home/dpaiton/IEDM/image_rram_encoding-"+str(img_id)+".npz", data=latent_encodings[img_id, ...])

#image_encoding  = np.load("/home/dpaiton/IEDM/image_rram_encoding-0.npz")["data"]
#import IPython; IPython.embed()
