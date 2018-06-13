import numpy as np
import tensorflow as tf
from data.nat_dataset import Dataset

params = {
  "data_shape": [128, 128, 1],
  "batch_size": 1,
  "num_preproc_threads": 8,
  "downsample_images": True,
  "downsample_method": "resize",
  "rand_seed": 123,
  "data_file":"/home/dpaiton/Work/DeepSparseCoding/broke_ass_imgs.txt"}
  #"data_file":"/media/tbell/datasets/test_images.txt"}
params["rand_state"] = np.random.RandomState(params["rand_seed"])
#/media/tbell/datasets/imagenet/ILSVRC/Data/DET/test/ILSVRC2013_test_00000475.JPEG
#/media/tbell/datasets/imagenet/ILSVRC/Data/DET/test/ILSVRC2013_test_00000476.JPEG
#/media/tbell/datasets/imagenet/ILSVRC/Data/DET/test/ILSVRC2013_test_00000477.JPEG
#/media/tbell/datasets/imagenet/ILSVRC/Data/DET/test/ILSVRC2013_test_00000478.JPEG
#/media/tbell/datasets/imagenet/ILSVRC/Data/DET/test/ILSVRC2013_test_00000479.JPEG

data = {"train": Dataset(params["data_file"], params)}

x_shape = [None,]+params["data_shape"]
w_shape = [8, 8, 1, 10]
test_graph = tf.Graph()
with tf.device("/gpu:0"):
  with test_graph.as_default():
    x = tf.placeholder(tf.float32, shape=x_shape, name="x")
    w_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
      seed=params["rand_seed"], dtype=tf.float32)
    w = tf.get_variable(name="w", shape=w_shape, dtype=tf.float32,
      initializer=w_init, trainable=True)
    y = tf.nn.conv2d(x, w, [1, 4, 4, 1], padding="SAME", use_cudnn_on_gpu=True, name="y")
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

num_epochs = 0
batch_counter = 0
while num_epochs < 1:
  data_batch = data["train"].next_batch(params["batch_size"])
  images = data_batch[0]
  locations = data_batch[1]
  print(batch_counter, "\n-------\n", locations, "\n-------\n")
  num_batches = data["train"].batches_completed
  num_epochs = data["train"].epochs_completed
  with tf.Session(config=config, graph=test_graph) as sess:
    sess.run(init_op,
      feed_dict={x:np.zeros([params["batch_size"]]+params["data_shape"], dtype=np.float32)})
    out_var = sess.run(y, feed_dict={x:images})
  batch_counter += 1