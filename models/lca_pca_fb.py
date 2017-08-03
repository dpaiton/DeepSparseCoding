import tensorflow as tf
import params.param_picker as pp
from models.lca_pca import LCA_PCA

class LCA_PCA_FB(LCA_PCA):
  def __init__(self, params, schedule):
    lca_params, lca_schedule = pp.get_params("lca")
    new_params = lca_params.copy()
    lca_pca_params, lca_pca_schedule = pp.get_params("lca_pca")
    new_params.update(lca_pca_params)
    new_params.update(params)
    super(LCA_PCA_FB, self).__init__(new_params, schedule)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    num_pooling_units [int] indicating the number of 2nd layer units
  """
  def load_params(self, params):
    super(LCA_PCA_FB, self).load_params(params)
    self.num_pooling_units = int(params["num_pooling_units"])
    self.fb_mult = int(params["fb_mult"])

  def infer_coefficients(self):
   lca_b = tf.matmul(self.x, self.phi, name="driving_input")
   lca_g = (tf.matmul(tf.transpose(self.phi), self.phi, name="gram_matrix")
     - tf.constant(np.identity(self.phi_shape[1], dtype=np.float32), name="identity_matrix"))
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.num_steps-1):
     u_list.append(self.step_inference(u_list[step], a_list[step], lca_b, lca_g))
     a_list.append(self.threshold_units(u_list[step+1]))
   return (u_list[-1], a_list[-1])

  def step_inference(self, u_in, a_in, b, g):
    with tf.name_scope("update_u") as scope:
      ## Get feedback component - add to inference term
      lca_explain_away = tf.matmul(a_in, g, name="explaining_away")
      du = tf.subtract(tf.subtract(b, lca_explain_away), u_in, name="du")
      u_out = tf.add(u_in, tf.multiply(self.eta, du))
    return u_out

