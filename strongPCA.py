import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import tensorflow as tf

import data.data_picker as dp
import utils.plot_functions as pf
import utils.image_processing as ip

params = {
  ## Model params
  "out_dir": os.path.expanduser("~")+"/Work/Projects/strongPCA/outputs/",
  "chk_dir": os.path.expanduser("~")+"/Work/Projects/strongPCA/checkpoints/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/",
  "load_chk": True,
  "update_interval": 100,
  "device": "/cpu:0",
  "learning_rate": 0.12,
  "num_neurons": 500,
  "sparse_mult": 0.2,
  "num_inference_steps": 20,
  "eta": 0.001/0.03, #dt/tau
  "eps": 1e-12,
  ## Data params
  "data_type": "vanhateren",
  "rand_state": np.random.RandomState(12345),
  "num_images": 5,#50,
  "num_batches": 30000,
  "batch_size": 100,
  "patch_edge_size": 16,
  "overlapping_patches": True,
  "patch_variance_threshold": 1e-6,
  "conv": False,
  "whiten_images": True}

## Calculated params
params["epoch_size"] = params["batch_size"] * params["num_batches"]
params["num_pixels"] = int(params["patch_edge_size"]**2)
params["dataset_shape"] = [int(val)
    for val in [params["epoch_size"], params["num_pixels"]]],
params["phi_shape"] = [params["num_pixels"], params["num_neurons"]]

graph = tf.Graph()
with tf.device(params["device"]):
  with graph.as_default():
    with tf.name_scope("placeholders") as scope:
      x = tf.placeholder(tf.float32, shape=[params["batch_size"],
        params["num_pixels"]], name="input_data")
      sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")

    with tf.name_scope("constants") as scope:
      u_zeros = tf.zeros(shape=tf.stack([tf.shape(x)[0],
        params["num_neurons"]]), dtype=tf.float32, name="u_zeros")
      u_noise = tf.truncated_normal(
        shape=tf.stack([tf.shape(x)[0], params["num_neurons"]]),
        mean=0.0, stddev=0.1, dtype=tf.float32, name="u_noise")

    with tf.name_scope("step_counter") as scope:
      global_step = tf.Variable(0, trainable=False, name="global_step")

    with tf.variable_scope("weights") as scope:
      phi_init = tf.truncated_normal(params["phi_shape"], mean=0.0, stddev=0.5,
        dtype=tf.float32, name="phi_init")
      phi = tf.get_variable(name="phi", dtype=tf.float32,
        initializer=phi_init, trainable=True)

    with tf.name_scope("norm_weights") as scope:
      norm_phi = phi.assign(tf.nn.l2_normalize(phi,
        dim=0, epsilon=params["eps"], name="row_l2_norm"))
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
        tf.constant(np.identity(params["phi_shape"][1], dtype=np.float32),
        name="identity_matrix"))
      lca_explain_away = tf.matmul(a, lca_g,
        name="explaining_away")
      du = lca_b - lca_explain_away - u
      step_inference = tf.group(u.assign_add(params["eta"] * du),
        name="step_inference")
      reset_activity = tf.group(u.assign(u_zeros),
        name="reset_activity")

    with tf.name_scope("performance_metrics") as scope:
      with tf.name_scope("reconstruction_quality"):
        MSE = tf.reduce_mean(tf.pow(tf.subtract(x, x_), 2.0),
          axis=[1, 0], name="mean_squared_error")

    with tf.name_scope("optimizers") as scope:
      learning_rates = tf.train.exponential_decay(
        learning_rate=params["learning_rate"],
        global_step=global_step,
        decay_steps=int(np.floor(params["num_batches"]*0.9)),
        decay_rate=0.5,
        staircase=True,
        name="phi_annealing_schedule")
      optimizer = tf.train.GradientDescentOptimizer(learning_rates,
        name="phi_optimizer")
      update_weights = optimizer.minimize(total_loss, global_step=global_step,
        var_list=[phi], name="phi_minimizer")

    full_saver = tf.train.Saver(var_list=[phi], max_to_keep=2)

    with tf.name_scope("summaries") as scope:
      #tf.summary.image("input", tf.reshape(x, [params["batch_size"],
      #  params["patch_edge_size"], params["patch_edge_size"], 1]))
      #tf.summary.image("weights", tf.reshape(tf.transpose(phi),
      #  [params["num_neurons"], params["patch_edge_size"],
      #  params["patch_edge_size"], 1]))
      tf.summary.histogram("u", u)
      tf.summary.histogram("a", a)
      tf.summary.histogram("phi", phi)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(params["out_dir"], graph)

    with tf.name_scope("initialization") as scope:
      init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

if not os.path.exists(params["out_dir"]):
  os.makedirs(params["out_dir"])
if not os.path.exists(params["chk_dir"]):
  os.makedirs(params["chk_dir"])

print("Loading data...")
data = dp.get_data(params["data_type"], params)
params["input_shape"] = [
  data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]

print("Initializing Session...")
with tf.Session(graph=graph) as sess:
  sess.run(init_op,
    feed_dict={x:np.zeros([params["batch_size"]]+params["input_shape"],
    dtype=np.float32)})

  if params["load_chk"]:
    full_saver.restore(sess, tf.train.latest_checkpoint(params["chk_dir"]))
  else:
    batch_steps = []
    losses = []
    sparsities = []
    recon_errors = []
    for b_step in range(params["num_batches"]):
      data_batch = data["train"].next_batch(params["batch_size"])
      input_data = data_batch[0]

      feed_dict = {x:input_data, sparse_mult:params["sparse_mult"]}

      sess.run(norm_weights, feed_dict)

      for inference_step in range(params["num_inference_steps"]):
        sess.run(step_inference, feed_dict)

      sess.run(update_weights, feed_dict)

      current_step = sess.run(global_step)
      if (current_step % params["update_interval"] == 0):
        summary = sess.run(merged_summaries, feed_dict)
        train_writer.add_summary(summary, current_step)
        full_saver.save(sess, save_path=params["chk_dir"]+"lca_chk",
          global_step=global_step)

        [current_loss, a_vals, recons, recon_err, weights] = sess.run(
          [total_loss, a, x_, MSE, phi], feed_dict)
        a_vals_max = np.array(a_vals.max()).tolist()
        a_frac_act = np.array(np.count_nonzero(a_vals)
          / float(params["batch_size"]*params["num_neurons"])).tolist()
        batch_steps.append(current_step)
        losses.append(current_loss)
        sparsities.append(a_frac_act)
        recon_errors.append(recon_err)

        print_dict = {"current_step":str(current_step).zfill(5),
          "loss":str(current_loss),
          "a_max":str(a_vals_max),
          "a_frac_act":str(a_frac_act)}
        print(print_dict)
        pf.save_data_tiled(weights.T.reshape((params["num_neurons"],
          params["patch_edge_size"], params["patch_edge_size"])),
          normalize=False, title="Dictionary at step "+str(current_step),
          save_filename=(params["out_dir"]+"phi_"+str(current_step).zfill(5)
          +".png"))
        pf.save_data_tiled(recons.reshape((params["batch_size"],
          params["patch_edge_size"], params["patch_edge_size"])),
          normalize=False, title="Recons at step "+str(current_step),
          save_filename=(params["out_dir"]+"recons_"
          +str(current_step).zfill(5)+".png"))
    output_data = {"batch_step":batch_steps, "total_loss":losses,
      "frac_active":sparsities, "recon_MSE":recon_errors}
    pf.save_stats(output_data, save_filename=(params["out_dir"]+"loss.png"))

  input_data = data["train"].next_batch(params["batch_size"])[0]
  feed_dict = {x:input_data, sparse_mult:params["sparse_mult"]}
  [a_vals, weights] = sess.run([a, phi], feed_dict)

  ## Testing
  pf.save_data_tiled(input_data.reshape((params["batch_size"],
    params["patch_edge_size"], params["patch_edge_size"])), normalize=False,
    title="Dim Reduced Data",
    save_filename=(params["out_dir"]+"dim_reduc_ALL_dat.png"))
  data_pca_reduc = ip.pca_reduction(input_data, num_dim=10)[0]
  pf.save_data_tiled(data_pca_reduc.reshape((params["batch_size"],
    params["patch_edge_size"], params["patch_edge_size"])), normalize=False,
    title="Dim Reduced Data",
    save_filename=(params["out_dir"]+"dim_reduc_10_dat.png"))
  data_pca_reduc = ip.pca_reduction(input_data, num_dim=100)[0]
  pf.save_data_tiled(data_pca_reduc.reshape((params["batch_size"],
    params["patch_edge_size"], params["patch_edge_size"])), normalize=False,
    title="Dim Reduced Data",
    save_filename=(params["out_dir"]+"dim_reduc_100_dat.png"))
  data_pca_reduc = ip.pca_reduction(input_data, num_dim=256)[0]
  pf.save_data_tiled(data_pca_reduc.reshape((params["batch_size"],
    params["patch_edge_size"], params["patch_edge_size"])), normalize=False,
    title="Dim Reduced Data",
    save_filename=(params["out_dir"]+"dim_reduc_256_dat.png"))

  (a_reduc, a_u, a_diagS, a_V) = ip.pca_reduction(a_vals, num_dim=24)

