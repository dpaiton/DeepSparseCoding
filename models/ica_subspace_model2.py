import numpy as np 
import tensorflow as tf
import untils.plot_functions as pf 
import utils.data_processing as dp 
from models.ica_model import IcaModel
from models.base_model import Model


class IcaSubspaceModel(Model):

    def __init__(IcaSubspaceModel):
        super(IcaSubspaceModel, self).__init__()

    def load_params(self, params):
        self.w_synth_shape = [self.num_neurons, self.params.num_pixels]
        self.w_analy_shape = [self.params.num_pixels, self.num_neurons]
        self.Q = construct_Q()

    def get_input_shape(self):
        return self.input_shape

    def build_graph_from_input(self, input_node):
        with tf.device(self.params.device):
            with tf.variable_scope("weights") as scope:
                Q, R = np.linalg.qr(np.random.standard_normal(self.w_analysis_shape), mode='complete')
                self.w_analy = tf.get_variable(name="w_analy",
                                               dtype=tf.float32,
                                               initializer=Q.astype(np.float32)),
                                               trainable=True)
                self.w_synth = tf.transpose(self.w_analy, name="w_analy")
                self.trainable_variables[self.w_analy.name] = self.w_analy

            with tf.variable_scope("inference") as scope:
                self.s = tf.matmul(self.w_synth, tf.transpose(input_node), name="latent_vars")

            with tf.variable_scope("log_liklihood") as scope:
                self.log_lik = self.compute_log_lik() 

            with tf.varaible_scope("output") as scope:
                self.recon = tf.matmul(self.w_synth, self.s, name="recon")

        self.graph_built = True

    def compute_w_grad(self, optimizer, weight_op=None):
        """Compute weight gradients"""
        def w_grad_per_input(I):
            wI = tf.matmul(I, self.w_analy)       # shape: (1, q)
            q = self.g(tf.matmul(tf.math.pow(wI, 2), self.Q)) # shape: (1, num_groups)
            nonlinear_term = tf.matmul(q, tf.transpose(self.Q)) # shape: (1, num_neurons)
            print("nonlinear_term":, nonlinear_term)
            print("wI", wI)
            scalars = tf.math.multiply(wI, nonlinear_term)
            tiled_I = tf.tile(I, [self.params.num_neurons, 1]) # shape: (num_neurons, num_pixels)
            w_gard = tf.transpose(tf.math.multiply(tiled_I, scalars)) # (num_piexls, num_neurons)
            return w_grad
        
        grad_list = tf.map_fn(w_grad_per_input(weight_op), self.input_placeholder)
        print("total gradients shape", total_gradients.shape)
        avg_grad = tf.reduce_mean(grad_list, axis=0)
        sum_grad = tf.reduce_sum(grad_list, axis=0)
        self.avg_grad = avg_grad
        self.sum_grad = sum_grad
        return [(avg_grad, weight_op)] # or sum grad

    def compute_log_lik(self):
        """Compute log liklihood. """
        det_term = self.params.batch_size * tf.linalg.det(self.w_analy)
        def per_img(I):
            wI = tf.matmul(self.w_analy, I)
            k = tf.matmul(tf.math.pow(wI, 2), self.Q)
            sun_k = tf.reduce_sum(self.log_p(k))
            return sum_k
        ll_list = tf.map_fn(per_img, self.input_placeholder)
        sum_ll = tf.reduce_sum(ll_list, axis=0)
        return sum_ll + det_term
        
    def log_p(x):
        """Probability Desnity. """
        return -1 * self.params.alpha * tf.math.pow(u, -0.5) + self.params.beta

    def g(x):
        """Non-linearity."""
        return -0.5 * self.params.alpha * tf.math.pow(u, -0.5)
    
    def construct_Q(self):
        Q = []
        for s, i in zip(self.group_sizes, self.group_index):
            col_index = np.zeros(self.params.num_neurons)
            col_index[i:i+s] = 1
            Q.append(col_index)
        Q = np.stack(Q, axis=1)
        Q = np.float32(Q)
        return Q

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(IcaSubspaceModel, self).generate_update_dict(input_data, input_labels, batch_step)
        feed_dict = self.get_feed_dict(input_data, input_labels)
        eval_list = [self.global_step, self.w_synth, self.w_analy, self.avg_grad, self.sum_grad, self.recon]
    
    def generate_plots(self, input_data, input_labels=None):
        pass

