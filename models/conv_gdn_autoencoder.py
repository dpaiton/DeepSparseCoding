import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_funcs as ef
import utils.get_data as get_data
import utils.mem_utils as mem_utils
from models.gdn_autoencoder import GDN_Autoencoder

class Conv_GDN_Autoencoder(GDN_Autoencoder):
  """
  Implementation of autoencoder described in Balle, Laparra, Simoncelli (2017)
  End-to-End Optimized Image Compression
  ## Key differences:
  #  Fully connected
  #  Single hidden layer, complete
  ## Methods ignored:
  #  add a small amount of uniform noise to input, to simulate pixel quantization
  """
  def __init__(self):
    super(Conv_GDN_Autoencoder, self).__init__()
    self.vector_inputs = False
    self.exclude_feed_strings = ["memristor_std_eps"]

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
    """
    super(GDN_Autoencoder, self).load_params(params)
    self.data_shape = params["data_shape"]
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(np.prod(self.data_shape))
    if "num_preproc_threads" in params.keys():
      self.num_preproc_threads = 1
    else:
      self.num_preproc_threads = int(params["num_preproc_threads"])
    # Dataset parameters
    self.batch_size = int(params["batch_size"])
    self.device = params["device"]
    self.downsample_images = params["downsample_images"]
    self.downsample_method = params["downsample_method"]
    # Loss parameters
    self.mle_step_size = float(params["mle_step_size"])
    self.num_mle_steps = int(params["num_mle_steps"])
    self.num_triangles = int(params["num_triangles"])
    self.sigmoid_beta = float(params["sigmoid_beta"])
    # Architecture parameters
    self.im_size_y = params["im_size_y"]
    self.im_size_x = params["im_size_x"]
    self.num_colors = params["num_colors"]
    self.patch_size_y = params["patch_size_y"] # list for encoding layers
    self.patch_size_x = params["patch_size_x"] # list for encoding layers
    self.input_channels = params["input_channels"] # list for encoding layers
    self.output_channels = params["output_channels"] # list for encoding layers
    self.w_strides = params["strides"] # list for encoding layers
    self.n_mem = self.compute_num_latent([self.im_size_y, self.im_size_x, self.num_colors],
      self.patch_size_y, self.patch_size_x, self.w_strides, self.output_channels)
    self.x_shape = [None, self.im_size_y, self.im_size_x, self.input_channels[0]]
    self.w_shapes = [vals for vals in zip(self.patch_size_y, self.patch_size_x,
      self.input_channels, self.output_channels)]
    self.w_shapes += self.w_shapes[::-1]
    self.w_strides += self.w_strides[::-1]
    self.b_shapes = [vals for vals in self.output_channels]
    # Decoding calculation uses conv2d_transpose so input_channels (w_shapes[2]) is actually
    # the number of output channels in the decoding direction
    self.w_gdn_shapes = [[val, val] for val in self.output_channels]
    self.w_gdn_shapes += [[val,val] for val in self.input_channels[::-1]]
    self.b_gdn_shapes = [[val] for val in self.output_channels]
    self.b_gdn_shapes += [[val] for val in self.input_channels[::-1]]
    self.input_channels += self.input_channels[::-1]
    self.output_channels += self.output_channels[::-1]
    self.num_layers = len(self.w_shapes)
    # Memristor parameters
    self.memristor_type = params["memristor_type"] # None indicates pasthrough
    self.memristor_data_loc = params["memristor_data_loc"]
    self.memristor_noise_shape = [self.batch_size, self.n_mem]
    self.mem_v_min = -1.0
    self.mem_v_max = 1.0

  def compute_num_latent(self, in_shape, patchY_list, patchX_list, strides, output_chans):
    inY, inX, inC = in_shape
    for patchY, patchX, out_chan, stride in zip(patchY_list, patchX_list, output_chans, strides):
      out_shape, num_out = self.compute_num_out([inY,inX,inC], [patchY,patchX],
        out_chan, [0,0], stride, stride)
      inY, inX, inC = out_shape
    return int(np.ceil(num_out))

  def compute_num_out(self, in_shape, patch_shape, num_features, pad_shape, strideX, strideY):
    inY, inX, inC = in_shape
    padY, padX = pad_shape
    patchY, patchX = patch_shape
    outX = 1 + (inX - patchX + 2 * padX) / strideX
    outY = 1 + (inY - patchY + 2 * padY) / strideY
    out_shape = [outY, outX, num_features]
    num_out = np.prod(out_shape)
    return (out_shape, num_out)

  def memristorize(self, u_in, memrister_std_eps, memristor_type=None):
    if memristor_type is None:
      return u_in
    elif memristor_type == "gauss":
      get_channel_data = get_data.get_gauss_data
    elif memristor_type == "rram":
      get_channel_data = get_data.get_rram_data
    elif memristor_type == "pcm":
      get_channel_data = get_data.get_pcm_data
    else:
      assert False, ("memristor_type must be None, 'rram', 'gauss', or 'pcm'")
    u_in_shape = tf.shape(u_in)
    (vs_data, mus_data, sigs_data,
      orig_VMIN, orig_VMAX, orig_RMIN,
      orig_RMAX) = get_channel_data(self.memristor_data_loc, self.n_mem, num_ext=5,
      norm_min=self.mem_v_min, norm_max=self.mem_v_max)
    v_clip = tf.clip_by_value(u_in, clip_value_min=self.mem_v_min,
      clip_value_max=self.mem_v_max)
    r = mem_utils.memristor_output(v_clip, self.memristor_std_eps, vs_data, mus_data, sigs_data,
      interp_width=np.array(vs_data[1, 0] - vs_data[0, 0]).astype('float32'))
    u_out = tf.reshape(r, shape=u_in_shape, name="mem_r")
    return u_out

  def compute_entropies(self, a_in):
    with tf.name_scope("unsupervised"):
      #TODO: Verify n_mem = prod(shape(a_in)[1:])
      num_units = self.n_mem#tf.reduce_prod(tf.shape(a_in)[1:])
      a_resh = tf.reshape(a_in, [self.batch_size, num_units])
      a_sig = self.sigmoid(a_resh, self.sigmoid_beta)
      a_probs = ef.prob_est(a_sig, self.mle_thetas, self.triangle_centers)
      a_entropies = tf.identity(ef.calc_entropy(a_probs), name="a_entropies")
    return a_entropies

  def compute_recon_loss(self, recon):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(recon.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(tf.subtract(self.x, recon)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_weight_decay_loss(self):
    with tf.name_scope("unsupervised"):
      decay_loss = tf.multiply(0.5*self.decay_mult,
        tf.add_n([tf.nn.l2_loss(weight) for weight in self.w_list]), name="weight_decay_loss")
    return decay_loss

  def compute_ramp_loss(self, a_in):
    ramp_loss = tf.reduce_mean(tf.reduce_sum(self.ramp_slope
      * (tf.nn.relu(a_in - self.mem_v_max)
      + tf.nn.relu(self.mem_v_min - a_in)), axis=[1,2,3]))
    return ramp_loss

  def get_loss_funcs(self):
    return {"recon_loss":self.compute_recon_loss,
      "entropy_loss":self.compute_entropy_loss,
      "weight_decay_loss":self.compute_weight_decay_loss,
      "ramp_loss":self.compute_ramp_loss}

  def compute_gdn_mult(self, layer_id, u_in, w_gdn, b_gdn, inverse):
    u_in_shape = tf.shape(u_in)
    w_min = 1e-3 # TODO: Make threshold a hyper parameter?
    w_threshold = tf.where(tf.less(w_gdn, tf.constant(w_min, dtype=tf.float32)),
      tf.multiply(w_min, tf.ones_like(w_gdn)), w_gdn)
    w_symmetric = tf.multiply(0.5, tf.add(w_threshold, tf.transpose(w_threshold)))
    b_min = 1e-3 # TODO: Make threshold a hyper parameter?
    b_threshold = tf.where(tf.less(b_gdn, tf.constant(b_min, dtype=tf.float32)),
      tf.multiply(b_min, tf.ones_like(b_gdn)), b_gdn)
    collapsed_u_sq = tf.reshape(tf.square(u_in),
      shape=tf.stack([u_in_shape[0]*u_in_shape[1]*u_in_shape[2], u_in_shape[3]]))
    weighted_norm = tf.reshape(tf.matmul(collapsed_u_sq, w_symmetric), shape=u_in_shape)
    gdn_mult = tf.sqrt(tf.add(weighted_norm, tf.square(b_threshold)))
    return gdn_mult

  def gdn(self, layer_id, u_in, inverse):
    """Devisive normalizeation nonlinearity"""
    with tf.variable_scope(self.weight_scope) as scope:
      if inverse:
        w_gdn = tf.get_variable(name="w_igdn"+str(layer_id), shape=self.w_gdn_shapes[layer_id],
          dtype=tf.float32, initializer=self.w_igdn_init, trainable=True)
        b_gdn = tf.get_variable(name="b_igdn"+str(layer_id), shape=self.b_gdn_shapes[layer_id],
          dtype=tf.float32, initializer=self.b_igdn_init, trainable=True)
      else:
        w_gdn = tf.get_variable(name="w_gdn"+str(layer_id), shape=self.w_gdn_shapes[layer_id],
          dtype=tf.float32, initializer=self.w_gdn_init, trainable=True)
        b_gdn = tf.get_variable(name="b_gdn"+str(layer_id), shape=self.b_gdn_shapes[layer_id],
          dtype=tf.float32, initializer=self.b_gdn_init, trainable=True)
    with tf.variable_scope("gdn"+str(layer_id)) as scope:
      gdn_mult = self.compute_gdn_mult(layer_id, u_in, w_gdn, b_gdn, inverse)
      if inverse:
        u_out = tf.multiply(u_in, gdn_mult, name="gdn_output"+str(layer_id))
      else:
        gdn_mult_min = 1e-6 # TODO: hyperparameter?
        u_out = tf.where(tf.less(gdn_mult, tf.constant(gdn_mult_min, dtype=tf.float32)), u_in,
          tf.divide(u_in, gdn_mult), name="gdn_output"+str(layer_id))
    return u_out, w_gdn, b_gdn, gdn_mult


  def layer_maker(self, layer_id, u_in, w_shape, w_stride, decode):
    """
    Make layer that does gdn(conv(u,w)+b)
    Note: In Balle et al:
      encoder layers compute u_out = gdn(conv(u_in,w)) i.e. linear-then-nonlinear transforms
      decoder layers compute u_out = conv(igdn(u_in),w) i.e. nonlinear-then-linear transforms
      Instead, we are going to have both layers compute u_out = [i]gdn(conv(u_in,w))
      i.e. linear-then-nonlinear transforms because this is what neural network people
      and comp neuro people typically do.
    """
    with tf.variable_scope(self.weight_scope) as scope:
      w = tf.get_variable(name="w"+str(layer_id), shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
    # Encode & Decode weights are the same shape for the corresponding layer
    # we use conv_transpose to transpose the feature dimension of the weights
    # therefore, bias needs to be size w_shape[3] for encoding and w_shape[2] for decoding
    bias_shape_id = 2 if decode else 3
    with tf.variable_scope(self.weight_scope) as scope:
      b = tf.get_variable(name="b"+str(layer_id), shape=w_shape[bias_shape_id],
        dtype=tf.float32, initializer=self.b_init, trainable=True)
    with tf.variable_scope("hidden"+str(layer_id)) as scope:
      if decode:
        height_const = 0 if u_in.get_shape()[1] % w_stride == 0 else 1
        out_height = (u_in.get_shape()[1] * w_stride) - height_const
        width_const = 0 if u_in.get_shape()[2] % w_stride == 0 else 1
        out_width = (u_in.get_shape()[2] * w_stride) - width_const
        out_shape = tf.stack([u_in.get_shape()[0], # Batch
          out_height, # Height
          out_width, # Width
          tf.constant(w_shape[2], dtype=tf.int32)]) # Channels
        conv_out = tf.add(tf.nn.conv2d_transpose(u_in, w, out_shape,
          strides = [1, w_stride, w_stride, 1], padding="SAME"), b,
          name="conv_out"+str(layer_id))
      else:
        conv_out = tf.add(tf.nn.conv2d(u_in, w, [1, w_stride, w_stride, 1],
          padding="SAME", use_cudnn_on_gpu=True), b, name="conv_out"+str(layer_id))
      #TODO: self.gdn_mult is going to get reset each layer - need to fix this for plotting
      gdn_out, w_gdn, b_gdn, self.gdn_mult = self.gdn(layer_id, conv_out, decode)
    return gdn_out, w, b, w_gdn, b_gdn, conv_out

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.triangle_centers = tf.placeholder(tf.float32, shape=[self.num_triangles],
            name="triangle_centers")
          self.ent_mult = tf.placeholder(tf.float32, shape=(), name="ent_mult")
          self.ramp_slope = tf.placeholder(tf.float32, shape=(), name="ramp_slope")
          self.decay_mult = tf.placeholder(tf.float32, shape=(), name="decay_mult")
          self.noise_var_mult = tf.placeholder(tf.float32, shape=(), name="noise_var_mult")

        with tf.name_scope("placeholders") as scope:
          self.memristor_std_eps = tf.placeholder(tf.float32, shape=self.memristor_noise_shape,
            name="memristor_std_eps")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("probability_estimate") as scope:
          self.mle_thetas, self.theta_init = ef.construct_thetas(self.n_mem, self.num_triangles)

        with tf.name_scope("weight_inits") as scope:
          self.w_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
            seed=self.rand_seed, dtype=tf.float32)
          self.b_init = tf.initializers.zeros(dtype=tf.float32)
          self.w_gdn_init = tf.initializers.random_uniform(minval=-1.0, maxval=1.0,
            dtype=tf.float32)
          self.b_gdn_init = tf.initializers.random_uniform(minval=1e-5, maxval=1.0,
            dtype=tf.float32)
          self.w_igdn_init = tf.initializers.random_uniform(minval=-1.0, maxval=1.0,
            dtype=tf.float32)
          self.b_igdn_init = tf.initializers.random_uniform(minval=1e-5, maxval=1.0,
            dtype=tf.float32)

        with tf.variable_scope("weights") as scope:
          self.weight_scope = tf.get_variable_scope()

        self.u_list = [self.x]
        self.conv_list = []
        self.w_list = []
        self.w_gdn_list = []
        self.b_list = []
        self.b_gdn_list = []
        for layer_id in range(self.num_layers):
          gdn_inverse = False if layer_id < self.num_layers/2 else True
          u_out, w, b, w_gdn, b_gdn, conv_out = self.layer_maker(layer_id, self.u_list[layer_id],
            self.w_shapes[layer_id], self.w_strides[layer_id], gdn_inverse)
          if layer_id == self.num_layers/2-1:
            #TODO: Verify n_mem = prod(shape(u_out)[1:])
            self.num_latent = self.n_mem#tf.reduce_prod(tf.shape(u_out)[1:], name="num_latent" )
            self.pre_mem = tf.identity(u_out, name="pre_memristor_activity")
            noise_var = tf.multiply(self.noise_var_mult, tf.subtract(tf.reduce_max(self.pre_mem),
              tf.reduce_min(self.pre_mem))) # 1/2 of 10% of range of gdn output
            uniform_noise = tf.random_uniform(shape=tf.stack(tf.shape(self.pre_mem)),
              minval=tf.subtract(0.0, noise_var), maxval=tf.add(0.0, noise_var))
            u_out = tf.add(uniform_noise, self.pre_mem, name="noisy_activity")
            u_out = self.memristorize(self.sigmoid(u_out, self.sigmoid_beta),
              self.memristor_std_eps, self.memristor_type)
          self.u_list.append(u_out)
          self.conv_list.append(conv_out)
          self.w_list.append(w)
          self.w_gdn_list.append(w_gdn)
          self.b_list.append(b)
          self.b_gdn_list.append(b_gdn)

        with tf.name_scope("inference") as scope:
          self.a = self.pre_mem

        with tf.variable_scope("probability_estimate") as scope:
          u_resh = tf.reshape(self.a, [self.batch_size, self.num_latent])
          u_sig = self.sigmoid(u_resh, self.sigmoid_beta)
          ll = ef.log_likelihood(u_sig, self.mle_thetas, self.triangle_centers)
          self.mle_update = [ef.mle(ll, self.mle_thetas, self.mle_step_size)
            for _ in range(self.num_mle_steps)]

        with tf.name_scope("loss") as scope:
          self.loss_dict = {"recon_loss":self.compute_recon_loss(self.u_list[-1]),
            "entropy_loss":self.compute_entropy_loss(self.a),
            "weight_decay_loss":self.compute_weight_decay_loss(),
            "ramp_loss":self.compute_ramp_loss(self.a)}
          self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

        with tf.name_scope("performance_metrics") as scope:
            with tf.name_scope("reconstruction_quality"):
              self.MSE = tf.reduce_mean(tf.square(tf.subtract(tf.multiply(self.u_list[0], 255.0),
                tf.multiply(tf.clip_by_value(self.u_list[-1], clip_value_min=-1.0,
                clip_value_max=1.0), 255.0))), reduction_indices=[1,2,3], name="mean_squared_error")
              self.batch_MSE = tf.reduce_mean(self.MSE, name="batch_mean_squared_error")
              self.SNRdB = tf.multiply(10.0, tf.log(tf.div(tf.square(tf.nn.moments(self.u_list[0],
                axes=[0,1,2,3])[1]), self.batch_MSE)), name="recon_quality")

        with tf.name_scope("summaries") as scope:
          tf.summary.image("input", self.u_list[0])
          tf.summary.image("reconstruction",self.u_list[-1])
          [tf.summary.histogram("u"+str(idx),u) for idx,u in enumerate(self.u_list)]
          [tf.summary.histogram("w"+str(idx),w) for idx,w in enumerate(self.w_list)]
          [tf.summary.histogram("w_gdn"+str(idx),w) for idx,w in enumerate(self.w_gdn_list)]
          [tf.summary.histogram("b"+str(idx),b) for idx,b in enumerate(self.b_list)]
          [tf.summary.histogram("b_gdn"+str(idx),u) for idx,u in enumerate(self.b_gdn_list)]
          tf.summary.scalar("total_loss", self.total_loss)
          tf.summary.scalar("batch_MSE", self.batch_MSE)
          tf.summary.scalar("SNRdB", self.SNRdB)
          self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.save_dir, self.graph)
    self.graph_built = True

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    mem_std_eps = np.random.standard_normal((self.params["batch_size"],
       self.n_mem)).astype(np.float32)
    feed_dict[self.memristor_std_eps] = mem_std_eps
    loss_list = [self.loss_dict[key] for key in self.loss_dict.keys()]
    eval_list = [self.global_step]+loss_list+[self.total_loss, self.a, self.u_list[-1]]
    init_eval_length = len(eval_list)
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step = out_vals[0]
    losses = out_vals[1:len(loss_list)+1]
    total_loss, a_vals, recon = out_vals[len(loss_list)+1:init_eval_length]
    input_mean = np.mean(input_data)
    input_max = np.max(input_data)
    input_min = np.min(input_data)
    recon_mean = np.mean(recon)
    recon_max = np.max(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_min = np.array(a_vals.min())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_min":a_vals_min,
      "a_fraction_active":a_frac_act,
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    for idx, key in enumerate(self.loss_dict.keys()):
      stat_dict[key] = losses[idx]
    grads = out_vals[init_eval_length:]
    for grad, name in zip(grads, grad_name_list):
      stat_dict[name+"_max_grad"] = learning_rate_dict[name]*np.array(grad.max())
      stat_dict[name+"_min_grad"] = learning_rate_dict[name]*np.array(grad.min())
    js_str = self.js_dumpstring(stat_dict)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    mem_std_eps = np.random.standard_normal((self.params["batch_size"],
       self.n_mem)).astype(np.float32)
    feed_dict[self.memristor_std_eps] = mem_std_eps
    eval_list = [self.global_step, self.a, self.u_list[int(self.num_layers/2-1)], self.w_list[0],
      self.w_list[-1], self.u_list[-1], self.gdn_mult]+self.w_gdn_list+self.b_gdn_list+self.b_list
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    assert np.all(np.stack([np.all(np.isfinite(arry)) for arry in eval_out])), (
      "Some plot evals had non-finite values")
    current_step = str(eval_out[0])
    pre_mem_activity, post_mem_activity, w_enc, w_dec, recon, gdn_mult = eval_out[1:7]
    w_gdn_list = eval_out[7:7+len(self.w_gdn_list)]
    b_gdn_list = eval_out[7+len(self.w_gdn_list):7+len(self.w_gdn_list)+len(self.b_gdn_list)]
    b_list = eval_out[7+len(self.w_gdn_list)+len(self.b_gdn_list):]
    w_enc_shape = w_enc.shape
    w_enc_norm = np.linalg.norm(w_enc.reshape([np.prod(w_enc_shape[:-1]), w_enc_shape[-1]]),
      axis=1, keepdims=False)
    w_dec = np.transpose(w_dec, axes=(0,1,3,2))
    w_dec_shape = w_dec.shape
    w_dec_norm = np.linalg.norm(w_dec.reshape([np.prod(w_dec_shape[:-1]), w_dec_shape[-1]]),
      axis=1, keepdims=False)

    #TODO:
    ##############
    #w_enc = np.transpose(w_enc, axes=(3,0,1,2))
    #w_enc = dp.reshape_data(w_enc, flatten=True)[0]
    #fig = pf.plot_data_tiled(w_enc, normalize=False,
    #  title="Encoding weights at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"w_enc_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    #w_dec = dp.reshape_data(w_dec, flatten=True)[0]
    #fig = pf.plot_data_tiled(w_dec, normalize=False,
    #  title="Decoding weights at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"w_dec_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    ##############

    fig = pf.plot_bar(w_enc_norm, num_xticks=5,
      title="w_enc l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"w_enc_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    fig = pf.plot_bar(w_dec_norm, num_xticks=5,
      title="w_dec l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"w_dec_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    for idx, w_gdn in enumerate(w_gdn_list):
      fig = pf.plot_weight_image(w_gdn, title="GDN "+str(idx)+" Weights", figsize=(10,10),
        save_filename=(self.disp_dir+"w_gdn_"+str(idx)+"_v"+self.version+"-"
        +current_step.zfill(5)+".png"))

    #TODO:
    ##############
    #for idx, b_gdn in enumerate(b_gdn_list):
    #  fig = pf.plot_activity_hist(b_gdn, title="GDN "+str(idx)+" Bias Histogram",
    #    save_filename=(self.disp_dir+"b_gdn_"+str(idx)+"_hist_v"+self.version+"-"
    #    +current_step.zfill(5)+".png"))
    #for idx, bias in enumerate(b_list):
    #  fig = pf.plot_activity_hist(bias, title="Bias "+str(idx)+" Histogram",
    #    save_filename=(self.disp_dir+"b_"+str(idx)+"_hist_v"+self.version+"-"
    #    +current_step.zfill(5)+".png"))
    #fig = pf.plot_activity_hist(gdn_mult, title="GDN Multiplier Histogram",
    #  save_filename=(self.disp_dir+"gdn_mult_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    ##############

    pre_mem_activity = dp.reshape_data(pre_mem_activity, flatten=True)[0]
    fig = pf.plot_activity_hist(pre_mem_activity, title="Activity Histogram (pre-mem)",
      save_filename=(self.disp_dir+"act_pre_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    post_mem_activity = dp.reshape_data(post_mem_activity, flatten=True)[0]
    fig = pf.plot_activity_hist(post_mem_activity, title="Activity Histogram (post-mem)",
      save_filename=(self.disp_dir+"act_post_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    input_data, input_orig_shape = dp.reshape_data(input_data, flatten=True)[:2]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=(self.disp_dir+"img_hist_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    input_data = dp.reshape_data(input_data, flatten=False, out_shape=input_orig_shape)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"images_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    recon = dp.reshape_data(recon, flatten=False)[0]
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
