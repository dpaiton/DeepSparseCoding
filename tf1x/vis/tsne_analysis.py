import os
import sys

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from scipy.misc import imsave

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)

import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap
import DeepSparseCoding.tf1x.utils.data_processing as dp

class tsne_params(object):
  model_type = "subspace_lca"
  model_name = "subspace_lca_mnist"
  version = "0.0"
  data_type = "mnist"
  input_scale = 0.5
  batch_size = 2500
  device = "/gpu:0"
  save_info = "analysis"
  eval_key = "inference/activity:0"
  overwrite_analysis = False
tsne_params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+tsne_params.model_name)

assert int(np.sqrt(tsne_params.batch_size))**2 == tsne_params.batch_size, (
  "batch_size parameter must have an even square root")

analyzer = ap.get_analyzer(tsne_params)

if not os.path.exists(analyzer.analysis_out_dir+"/embedding"):
  os.makedirs(analyzer.analysis_out_dir+"/embedding")

# Load natural image data
analyzer.model_params.patch_variance_threshold = 1e-5
data = ds.get_data(analyzer.model_params)
data = analyzer.model.preprocess_dataset(data, analyzer.model_params)
data = analyzer.model.reshape_dataset(data, analyzer.model_params)
analyzer.model_params.data_shape = list(data["train"].shape[1:])
analyzer.model.setup(analyzer.model_params, analyzer.model_schedule)
analyzer.model_params.input_shape = [data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]
raw_data_batch = data["train"].next_batch(tsne_params.batch_size)[0] # image data

# Load gabor data
#analyzer.model_params.data_shape = [256]
#analyzer.model.setup(analyzer.model_params, analyzer.model_schedule)
#analyzer.model_params.input_shape = [256]
#gabors = pickle.load(open("./random_gabor_stim.p", "rb"))
#raw_data_batch = gabors[:tsne_params.batch_size].reshape(tsne_params.batch_size, 256)

# Preprocess data
if hasattr(analyer.model_params, "whiten_data") and analyzer.model_params.whiten_data:
  data_batch, data_pre_wht_mean, data_wht_filter = \
    dp.whiten_data(raw_data_batch, method=analyzer.model_params.whiten_method)
if hasattr(analyzer.model_params, "lpf_data") and analyzer.model_params.lpf_data:
  data_batch, data_pre_lpf_mean, data_lp_filter = \
    dp.lpf_data(raw_data_batch, cutoff=analyzer.model_params.lpf_cutoff)
raw_data_batch, orig_shape, num_examples, num_rows, num_cols = dp.reshape_data(raw_data_batch,
  flatten=False)[:5]
#data_batch = tsne_params.input_scale * (data_batch / np.max(np.abs(data_batch)))
data_batch = raw_data_batch
assert num_rows == num_cols, ("The data samples must be square")

data_shape = data_batch.shape
imgs_per_edge = int(np.sqrt(tsne_params.batch_size))
sprite_edge_size = imgs_per_edge * num_rows

#reformat data into a tiled image for visualization
sprite = data_batch.reshape(((imgs_per_edge, imgs_per_edge)+data_shape[1:]))
sprite = sprite.transpose(((0,2,1,3) + tuple(range(4, data_batch.ndim+1))))
sprite = sprite.reshape((sprite_edge_size, sprite_edge_size)+sprite.shape[4:])
sprite = dp.rescale_data_to_one(sprite)[0]
sprite = 255 * sprite
sprite = sprite.astype(np.uint8)
sprite_out_dir = analyzer.analysis_out_dir+"/embedding/sprite_"+tsne_params.save_info+".png"
imsave(sprite_out_dir, sprite.squeeze())

input_data = dp.reshape_data(data_batch, flatten=True)[0]
#latent_representation = analyzer.evaluate_model(input_data, var_names=[tsne_params.eval_key])
latent_representation = analyzer.compute_pooled_activations(input_data)

tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()
embedding_var = tf.Variable(latent_representation, name="image_embedding") # Model embedding
#embedding_var = tf.Variable(latent_representation[tsne_params.eval_key], name="image_embedding") # Model embedding
#embedding_var = tf.Variable(input_data, name="image_embedding") # Identity embedding
tf.compat.v1.global_variables_initializer().run()

saver = tf.compat.v1.train.Saver()
summary_writer = tf.compat.v1.summary.FileWriter(analyzer.analysis_out_dir+"/embedding/")

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.sprite.image_path = sprite_out_dir
embedding.sprite.single_image_dim.extend([num_rows, num_cols])

projector.visualize_embeddings(summary_writer, config)

saver.save(sess, os.path.join(analyzer.analysis_out_dir+"/embedding/embedding_model.ckpt"))

summary_writer.close()
