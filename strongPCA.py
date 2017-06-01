import os

def get_dataset(num_images, out_shape):
  img_filename = (os.path.expanduser("~")
    +"/Work/Datasets/vanHateren/img/images_curated.h5")
  full_img_data = extract_images(img_filename, num_images)
  full_img_data = ip.downsample_data(full_img_data, factor=[1, 0.5, 0.5],
    order=2)
  full_img_data = ip.standardize_data(full_img_data)
  dataset = ip.extract_patches(full_img_data, out_shape, True, 1e-6)
  return dataset

def extract_images(filename, num_images=50):
  with h5py.File(filename, "r") as f:
    full_img_data = np.array(f["van_hateren_good"], dtype=np.float32)
    im_keep_idx = np.random.choice(full_img_data.shape[0], num_images,
      replace=False)
    return full_img_data[im_keep_idx, ...]

## Model params
out_dir = os.path.expanduser("~")+"/Work/StrongPCA/outputs/"

update_interval = 500
device = "/cpu:0"
num_batches = 30000
batch_size = 100
learning_rate = 0.01
num_neurons = 500

## Data params
num_images = 50
epoch_size = int(1e6)
patch_edge_size = 16
whiten = True

## Calculated params
num_pixels = int(patch_edge_size**2)
dataset_shape = [int(val)
  for val in [epoch_size, num_pixels]]
phi_shape = [num_pixels, num_neurons]

graph = tf.Graph()
with tf.device(device):
  with graph.as_default():
    with tf.name_scope("placeholders") as scope:
      x = tf.placeholder(tf.float32, shape=[batch_size, num_pixels],
        name="input_data")
      sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")

    with tf.name_scope("constants") as scope:
      u_zeros = tf.zeros(shape=tf.stack([tf.shape(x)[0], num_neurons]),
        dtype=tf.float32, name="u_zeros")
      u_noise = tf.truncated_normal(
        shape=tf.stack([tf.shape(x)[0], num_neurons]),
        mean=0.0, stddev=0.1, dtype=tf.float32, name="u_noise")

    with tf.name_scope("step_counter") as scope:
      global_step = tf.Variable(0, trainable=False, name="global_step")

    with tf.variable_scope("weights") as scope:
      phi_init = tf.truncated_normal(phi_shape, mean=0.0, stddev=0.5,
        dtype=tf.float32, name="phi_init")
      phi = tf.get_variable(name="phi", dtype=tf.float32,
        initializer=phi_init, trainable=True)

    with tf.name_scope("norm_weights") as scope:
      norm_phi = phi.assign(tf.nn.l2_normalize(phi,
        dim=0, epsilon=eps, name="row_l2_norm"))
      norm_weights = tf.group(norm_phi,
        name="l2_normalization")

    with tf.name_scope("inference") as scope:
      u = tf.Variable(u_zeros, trainable=False,
        validate_shape=False, name="u")
      # soft thresholded, rectified
      a = tf.where(tf.greater(u, sparse_mult),
        tf.subtract(u, sparse_mult), u_zeros,
        name="activity")

    with tf.name_scope("output") as scope:
      with tf.name_scope("image_estimate"):
        x_ = tf.matmul(a, tf.transpose(phi),
          name="reconstruction")

    with tf.name_scope("loss") as scope:
      with tf.name_scope("unsupervised"):
        recon_loss = tf.reduce_mean(0.5 *
          tf.reduce_sum(tf.pow(tf.subtract(x, x_), 2.0),
          axis=[1]), name="recon_loss")
        sparse_loss = sparse_mult * tf.reduce_mean(
          tf.reduce_sum(tf.abs(a), axis=[1]),
          name="sparse_loss")
        unsupervised_loss = (recon_loss + sparse_loss)
      total_loss = unsupervised_loss

    with tf.name_scope("update_u") as scope:
      lca_b = tf.matmul(x, phi, name="driving_input")
      lca_g = (tf.matmul(tf.transpose(phi), phi,
        name="gram_matrix") -
        tf.constant(np.identity(phi_shape[1], dtype=np.float32),
        name="identity_matrix"))
      lca_explain_away = tf.matmul(a, lca_g,
        name="explaining_away")
      du = lca_b - lca_explain_away - u
      step_inference = tf.group(u.assign_add(eta * du),
        name="step_inference")
      reset_activity = tf.group(u.assign(u_zeros),
        name="reset_activity")

    with tf.name_scope("performance_metrics") as scope:
      with tf.name_scope("reconstruction_quality"):
        MSE = tf.reduce_mean(tf.pow(tf.subtract(x, x_), 2.0),
          axis=[1, 0], name="mean_squared_error")

    with tf.name_scope("optimizers") as scope:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate,
        name="phi_optimizer")
      update_weights = optimizer.minimize(total_loss, global_step=global_step,
        var_list=[phi], name="phi_minimizer")

    full_saver = tf.train.Saver(var_list=[phi], max_to_keep=2)

    with tf.name_scope("summaries") as scope:
      #tf.summary.image("input", tf.reshape(x, [batch_size, patch_edge_size,
      #  patch_edge_size, 1]))
      #tf.summary.image("weights", tf.reshape(tf.transpose(phi), [num_neurons,
      #  patch_edge_size, patch_edge_size, 1]))
      tf.summary.histogram("u", u)
      tf.summary.histogram("a", a)
      tf.summary.histogram("phi", phi)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(out_dir, graph)

    with tf.name_scope("initialization") as scope:
      init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

