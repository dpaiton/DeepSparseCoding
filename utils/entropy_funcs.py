import tensorflow as tf

def construct_thetas(num_latent, num_tri):
    theta_init = tf.truncated_normal((num_latent, num_tri), mean=1.0, stddev=0.01,
      dtype=tf.float32, name="theta_init")
    return (tf.Variable(initial_value=theta_init, name="thetas"), theta_init)

def weights(thetas):
    return tf.exp(thetas, name="weights")

def zeta(thetas):
    """
    Normalizing constant
    Input:
        thetas [num_latent, num_tri]
    Output:
        zeta [num_latent]
    """
    return tf.reduce_sum(weights(thetas), axis=[1], name="zetas")

def eval_triangle(x, h, n):
    """
    Compute triangle histogram for given latent variables
    Input:
        x [num_batch, num_latent] latent values
        h [num_latent, num_tri] triangle heights
        n [num_tri] number of triangles to use
    x is broadcasted to [num_batch, num_latent, num_tri] (replicated num_tri times)
    h is broadcasted to [num_batch, num_latent, num_tri] (replicated num_batch times)
    n is boradcasted to [num_batch, num_latent, num_tri] (replicated num_batch * num_latent times)
    Output:
        y [num_batch, num_latent, num_tri] evaluated triangles
    """
    x = tf.expand_dims(x, axis=-1) # [num_batch, num_latent, 1]
    h = tf.expand_dims(h, axis=0) # [1, num_latent, num_tri]
    n = tf.expand_dims(tf.expand_dims(n, axis=0), axis=0) # [1, 1, num_tri]
    y = tf.nn.relu(tf.subtract(h, tf.abs(tf.subtract(x, n,
      name="n_sub"), name="abs_shifted"), name="h_sub"), name="tri_out")
    return y

def prob_est(latent_vals, thetas, tri_locs):
    """
    Inputs:
        latent_vals [num_batch, num_latent] latent values
        thetas [num_latent, num_tri] triangle weights
        tri_locs [num_tri] location of each triangle for latent discretization
    Outputs:
        prob_est [num_batch, num_latent]
    """
    tris = eval_triangle(latent_vals, weights(thetas), tri_locs) # [num_batch, num_latent, num_tri]
    prob_est = tf.divide(tf.reduce_sum(tris, axis=[2], name="tris_reduced"),
        1e-9+tf.expand_dims(zeta(thetas), axis=0, name="expanded_zeta"), name="prob_est")
    return prob_est

def safe_log(probs, eps=1e-9):
  logprob = tf.where(tf.less_equal(probs, tf.zeros_like(probs)+eps, name="prob_le_zero"),
    tf.zeros_like(probs), tf.log(probs, name="log_prob"), name="safelog_where")
  return logprob

def log_likelihood(latent_vals, thetas, tri_locs):
    """
    Inputs:
        latent_vals [num_batch, num_latent] latent values
        thetas [num_latent, num_tri] triangle weights
        tri_locs [num_tri] location of each triangle for latent discretization
    Outputs:
        log_likelihood [num_latent]
    """
    probs = prob_est(latent_vals, weights(thetas), tri_locs) # [num_batch, num_latent]
    logprobs = safe_log(probs)
    return tf.reduce_sum(logprobs, axis=[0], name="log_likelihood")

def mle(log_likelihood, thetas, learning_rate):
    grads = tf.gradients(log_likelihood, thetas, name="ll_grads")[0]
    op = thetas.assign_add(tf.multiply(tf.constant(learning_rate), grads, name="rescale_grad"))
    return op

def calc_entropy(probs):
    """
    Inputs:
        probs [num_batch, num_latent]
    Outputs:
        entropy [num_latent]
    """
    plogp = tf.multiply(probs, safe_log(probs), name="plogp")
    return -tf.reduce_sum(plogp, axis=[0], name="entropy")
