import numpy as np 
import tensorflow as tf 
import utils.plot_functions as pf 
import utils.data_processing as dp 
from models.ica_model import IcaModel

class IcaSubspaceModel(IcaModel):

  def __init__(self):
    super(IcaSubspaceModel, self).__init__()

  def load_params(self, params):
    """Parameters are the same as classical ICA, with addition of number of subspace/groups and
    group sizes for each subspace.
    
    Variables:
      num_groups: (int) number of groups/subspaces
      group_sizes: (int[]) number of vectors at each subspace. ith index for ith group.
      group_index: (int[]) index of vectors for each subspace
    

    """
    super(IcaSubspaceModel, self).load_params(params)
    self.input_shape = [None, self.num_neurons]

    # new params for subspace ica
    self.num_groups = self.params.num_groups
    self.group_sizes = self.params.group_sizes
    #self.group_sizes = self.construct_group_sizes(self.params.group_sizes)
    self.group_index = [sum(self.group_sizes[:i]) for i in range(self.num_groups)]
    self.sum_arr = self.construct_sum_arr() 


  def get_input_shape(self):
      return self.input_shape


  def build_graph_from_input(self, input_node):
    """Build the Tensorflow graph object. 

    Placeholders:
      input_img: (float[]) input image patch

    Variables:
      w_synth: (float[][]) synthesis weights; basis; "A" in classical ICA
      w_analy: (float[][]) analysis weights; inverse basis; "A.T" in classical ICA (since A is orthonormal)
      s: (float[]) latent variables, computed from inner product of w_analy and input_img
      recon: (float[]) recontruction of image patch using w_synth and s

    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        self.input_img = input_node

        with tf.variable_scope("weights") as scope:
          Q, R = np.linalg.qr(np.random.standard_normal(self.w_analysis_shape), mode="complete")
          #w_init = np.eye(self.params.num_neurons)[:, 48:self.params.num_neurons-48]
          self.w_analy = tf.get_variable(name="w_analy", 
                                         dtype=tf.float32, 
                                         initializer=Q.astype(np.float32), 
                                         trainable=True)
 #         print(self.w_synth.shape)
#          orthonormalize = lambda x: tf.linalg.qr(x, full_matrices=True, name="w_synth")[0]
#          self.w_synth = tf.map_fn(orthonormalize, self.w_synth, name="orthnormal_w_synth")
          self.w_synth = tf.transpose(self.w_analy, name="w_synth")
#          self.w_synth = tf.matrix_inverse(self.w_analy, name="w_synth")
          self.trainable_variables[self.w_analy.name] = self.w_analy
#          self.trainable_variables[self.w_analy.name] = self.w_analy

        with tf.name_scope("inference") as scope:
          self.s = tf.matmul(self.w_synth, tf.transpose(input_node), name="latent_variables")
          self.a = tf.identity(self.s, name="activity")
          self.z = tf.sign(self.a)
         

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.reconstruction = tf.matmul(self.w_synth, self.s, name="reconstruction")

    self.graph_built = True

  def compute_single_weight_gradients(self, weight_op):
    def nonlinearity(u, alpha=1):
        return -1 * alpha * tf.math.pow(u, -0.5)
    
    def compute_grad(img_patch):
        img_patch = tf.reshape(img_patch, [self.params.num_pixels, 1])
        wI = tf.transpose(tf.matmul(tf.transpose(img_patch), weight_op[0]))

        group_scalars = tf.matmul(tf.transpose(tf.math.pow(wI, 2)), self.sum_arr)
        nonlinear_term = tf.matmul(self.sum_arr, nonlinearity(tf.transpose(group_scalars)))
        scalars = tf.math.multiply(wI, nonlinear_term)
        img_tiled = tf.tile(img_patch, [1, self.num_neurons])
        gradient = tf.transpose(tf.multiply(tf.transpose(img_tiled), scalars))
        return gradient
    return compute_grad


  def compute_weight_gradients1(self, optimizer, weight_op=None):
        if(type(weight_op) is not list):
            weight_op = [weight_op]

        assert len(weight_op) == 1, ("IcaModel should only have one weight matrix")

        gradients = []
        for i in range(self.params.batch_size):
            img = tf.slice(self.input_img, [i, 0], [1, self.params.patch_edge_size*self.params.patch_edge_size])
            gradients.append(self.compute_single_weight_gradients(weight_op)(img))
#        gradients = tf.map_fn(self.compute_single_weight_gradients(weight_op), self.input_img)
        avg_gradient = - tf.reduce_mean(gradients, axis=0)
        self.avg_grad = avg_gradient
        return [(avg_gradient, weight_op[0])]

  def compute_weight_gradients(self, optimizer, weight_op=None):

    def nonlinearity(u, alpha=1):
        return -1 * alpha * tf.math.pow(u, -0.5)
 
    p = []
    for j in range(64):
        for _ in range(4):
            p.append((j*4, (j*4)+4))

    weight_grads = []
    for img_i in range(self.params.batch_size):
        img = tf.slice(self.input_img, [img_i, 0], [1, 256])
        img = tf.reshape(img, (256, 1))

        wI = tf.matmul(tf.transpose(weight_op), img)
        wI2 = tf.pow(wI, 2)

        weight_mat = []
        for w_i in range(self.params.num_pixels):
            g1, g2 = p[w_i][0], p[w_i][1]
            inner_prod_term = wI[w_i][0]
            nonlinear_term = nonlinearity(tf.reduce_sum(wI2[g1:g2]))
            
            img = tf.reshape(img, [-1])
            weight_mat.append(img * inner_prod_term * nonlinear_term)
        weight_mat = tf.stack(weight_mat, axis=0)
        weight_grads.append(weight_mat)
        
    avg_grad = - tf.reduce_mean(weight_grads, axis=0) 
    self.avg_grad = avg_grad

    return [(avg_grad, weight_op)]
        

          



  def construct_group_sizes(self, params_group_sizes):
    """Construct respective group sizes. If group_size initialzed as None, then group sizes are uniformally
    distributed; unless specified otherwise. """
    self.group_sizes = params_group_sizes
    if params_group_sizes is None:
      self.group_sizes = [self.num_neurons // self.num_groups for _ in range(self.num_groups)]
    
    assert sum(self.group_sizes) == self.num_neurons, ("Total number of vectors should be the same "
                                                        "as number of neurons.")
    print("construct_group_sizes: {}".format(self.group_sizes))
    return self.group_sizes

  def construct_sum_arr(self):
    sum_arr = []
    for s, i in zip(self.group_sizes, self.group_index):
      col_index = np.zeros(self.num_neurons)
      col_index[i:i+s] = 1
      sum_arr.append(col_index)
    sum_arr = np.stack(sum_arr, axis=1)
    sum_arr = np.float32(sum_arr)
    return sum_arr

  def get_subspace(self, w, g):
    """Return the column vectors in the g-th subspace. """
    num_vec = self.group_sizes[g]
    subspace_index = self.group_index[g]
    return w[:, subspace_index:subspace_index+num_vec]

  def orthonormalize(self, weight, norm=False, ortho=False):
      if norm:
          weight = weight / tf.norm(weight, ord=2, axis=0)  
      q, r = tf.linalg.qr(weight, full_matrices=True)
      return q

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    update_dict = super(IcaSubspaceModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w_synth, self.w_analy, self.avg_grad, self.s, self.reconstruction]
    stat_dict = {}

    # evaluate values
    out_vals = tf.get_default_session().run(eval_list, feed_dict)
    global_step, w_synth, w_analy, avg_grad, latent_vars, recon = out_vals

    # meta info
    stat_dict["global_step"] = global_step
    stat_dict["batch_step"] = batch_step 

    # weights
    stat_dict["w_synth"] = w_synth
    stat_dict["w_analy"] = w_analy
    stat_dict["avg_grad"] = avg_grad
    print("w_synth: \n")
    print(w_synth)
    print("avg_grad\n")
    print(avg_grad)

    
    # activations
    stat_dict["latent_vars"] = latent_vars

    # get subspace and store them in a list
    stat_dict["group_sizes"] = self.group_sizes
    stat_dict["group_index"] = self.group_index
    stat_dict["sum_arr"] = self.sum_arr
#    subspaces = []
#    print("group_index: {}".format(self.group_index))
#    for g_i in self.group_index: 
#        subspaces.append(self.get_subspace(g_i))
#    stat_dict["subspaces"].append(subspaces)

    update_dict.update(stat_dict)
    return update_dict


  def generate_plots(self, input_data, input_labels=None):
    super(IcaModel, self).generate_plots(input_data, input_labels)
    ## ADD FUCNITONS
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w_synth, self.w_analy, self.s]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    
    # step
    current_step = str(eval_out[0])

    # weights
    w_synth_eval, w_analy_eval = eval_out[1], eval_out[2]
    weight_shape = [self.num_neurons, 
                    int(np.sqrt(self.params.num_pixels)), 
                    int(np.sqrt(self.params.num_pixels))]
    w_synth_eval = np.reshape(w_synth_eval, weight_shape)
    w_analy_eval = np.reshape(w_analy_eval, weight_shape)
    pf.plot_weights(w_synth_eval,
                    save_filename="{}w_synth_eval_{}.png".format(self.params.display_dir, current_step))
    pf.plot_weights(w_analy_eval,
                    save_filename="{}w_analy_eval_{}.png".format(self.params.display_dir, current_step))
    
    # groups
    subspaces = []
    #print("group_index: {}".format(self.group_index))
    #for g_i in self.group_index: 
    #    subspaces.append(self.get_subspace(w_analy_eval, g_i))
#   
    
    # activations
    latent_eval = eval_out[2]
    
    

  
    

    

    
