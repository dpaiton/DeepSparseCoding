import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from scipy.misc import imsave
import data.data_selector as ds
import analysis.analysis_picker as ap
import utils.data_processing as dp

tsne_params = {
  "model_type": "lca",
  #"model_type": "ica",
  #"model_type": "sparse_autoencoder",
  "model_name": "lca_256_l0_2.5",
  #"model_name": "ica_test",
  #"model_name": "sparse_autoencoder",
  "version": "1.0",
  #"version": "0.0",
  #"version": "0.0",
  "data_type": "vanHateren",
  "input_scale": 40, # rescale input to be [-input_scale, input_scale]; 40 for LCA, 0.5 for ICA
  #"input_scale": 0.5, # rescale input to be [-input_scale, input_scale]; 40 for LCA, 0.5 for ICA
  "batch_size": 1024,
  "device": "/gpu:0",
  "save_info": "analysis",
  "overwrite_analysis": False}
tsne_params["model_dir"] = (os.path.expanduser("~")+"/Work/Projects/"+tsne_params["model_name"])

assert int(np.sqrt(tsne_params["batch_size"]))**2 == tsne_params["batch_size"], (
  "batch_size parameter must have an even square root")

analyzer = ap.get_analyzer(tsne_params)

analyzer.model_params["patch_variance_threshold"] = 1e-5
#data = ds.get_data(analyzer.model_params)
#data = analyzer.model.preprocess_dataset(data, analyzer.model_params)
#data = analyzer.model.reshape_dataset(data, analyzer.model_params)

analyzer.model_params["data_shape"] = [256]#list(data["train"].shape[1:])
analyzer.model.setup(analyzer.model_params, analyzer.model_schedule)
analyzer.model_params["input_shape"] = [256]#[data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]

if not os.path.exists(analyzer.analysis_out_dir+"/embedding"):
  os.makedirs(analyzer.analysis_out_dir+"/embedding")

#data_batch = data["train"].next_batch(tsne_params["batch_size"])[0]
gabors = pickle.load(open("./random_gabor_stim.p", "rb"))
raw_data_batch = gabors[:tsne_params["batch_size"]].reshape(tsne_params["batch_size"], 256)
raw_data_batch, orig_shape, num_examples, num_rows, num_cols = dp.reshape_data(raw_data_batch,
  flatten=False)[:5]
if "whiten_data" in analyzer.model_params.keys() and analyzer.model_params["whiten_data"]:
  data_batch, data_pre_wht_mean, data_wht_filter = \
    dp.whiten_data(raw_data_batch, method=analyzer.model_params["whiten_method"])
if "lpf_data" in analyzer.model_params.keys() and analyzer.model_params["lpf_data"]:
  data_batch, data_pre_lpf_mean, data_lp_filter = \
    dp.lpf_data(raw_data_batch, cutoff=analyzer.model_params["lpf_cutoff"])
data_batch = tsne_params["input_scale"] * (data_batch / np.max(np.abs(data_batch)))
assert num_rows == num_cols, ("The data samples must be square")

data_shape = data_batch.shape
imgs_per_edge = int(np.sqrt(tsne_params["batch_size"]))
sprite_edge_size = imgs_per_edge * num_rows

#reformat data into a tiled image for visualization
sprite = data_batch.reshape(((imgs_per_edge, imgs_per_edge)+data_shape[1:]))
sprite = sprite.transpose(((0,2,1,3) + tuple(range(4, data_batch.ndim+1))))
sprite = sprite.reshape((sprite_edge_size, sprite_edge_size)+sprite.shape[4:])
sprite = dp.rescale_data_to_one(sprite)
sprite = 255 * sprite
sprite = sprite.astype(np.uint8)
sprite_out_dir = analyzer.analysis_out_dir+"/embedding/sprite_"+tsne_params["save_info"]+".png"
imsave(sprite_out_dir, sprite.squeeze())

input_data = dp.reshape_data(data_batch, flatten=True)[0]
latent_representation = analyzer.evaluate_model(input_data, var_names=["inference/activity:0"])

tf.reset_default_graph()
sess = tf.InteractiveSession()
embedding_var = tf.Variable(latent_representation["inference/activity:0"], name="image_embedding")
tf.global_variables_initializer().run()

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(analyzer.analysis_out_dir+"/embedding/")

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
#embedding.metadata_path = os.path.join(analyzer.analysis_out_dir+"/embedding/metadata.tsv")
embedding.sprite.image_path = sprite_out_dir
embedding.sprite.single_image_dim.extend([num_rows, num_cols])

projector.visualize_embeddings(summary_writer, config)

saver.save(sess, os.path.join(analyzer.analysis_out_dir+"/embedding/embedding_model.ckpt"))

summary_writer.close()
