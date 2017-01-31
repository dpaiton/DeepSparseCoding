import matplotlib
matplotlib.use("Agg")

## TODO:
##  why is db always the same shape?
##  Estimate kurtosis (q in K&L paper) from layer 1 activity using EM
##  specify parameter that allows you to load in "phi" and set it for "a"
##   Will probably require you to load in the original model, eval "phi",
##   then assign it to a constant for "a"

import numpy as np
import tensorflow as tf
import models.model_picker as mp
from data.MNIST import load_MNIST

## Import parameters & schedules
#from mlp_params import params, schedule
from karklin_params import params, schedule

## Get data
np_rand_state = np.random.RandomState(params["rand_seed"])
data = load_MNIST(params["data_dir"], normalize_imgs=params["norm_images"],
  rand_state=np_rand_state)

model = mp.get_model(params, schedule)

model.write_saver_defs()

with tf.Session(graph=model.graph) as sess:
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros((model.num_pixels, model.batch_size),
    dtype=np.float32)})

  model.write_graph(sess.graph_def)

  for sch_idx, sch in enumerate(schedule):
    model.sched_idx = sch_idx
    model.log_info("Beginning schedule "+str(sch_idx))
    for b_step in range(model.get_sched("num_batches")):
      mnist_batch = data["train"].next_batch(model.batch_size)
      input_images = mnist_batch[0].T

      feed_dict = model.get_feed_dict(input_images)

      ## Normalize weights
      if params["norm_weights"]:
        sess.run(model.normalize_weights)

      ## Clear activity from previous batch
      sess.run([model.clear_u, model.clear_v], feed_dict)

      ## Run inference
      if hasattr(model, "do_inference"):
        sess.run([model.do_inference], feed_dict)

      ## Update weights
      for w_idx in range(len(model.get_sched("weights"))):
        sess.run(model.apply_grads[sch_idx][w_idx], feed_dict)

      ## Generate logs
      current_step = sess.run(model.global_step)
      if (current_step % model.log_int == 0
        and model.log_int > 0):
        model.print_update(input_data=input_images, input_label=None,
          batch_step=b_step+1)

      ## Plot weights & gradients
      if (current_step % model.gen_plot_int == 0
        and model.gen_plot_int > 0):
        model.generate_plots(input_data=input_images)

      ## Checkpoint
      if (current_step % model.cp_int == 0
        and model.cp_int > 0):
        save_dir = model.write_checkpoint(sess)

      ## Adjust the feedback influence
      ## TODO: Don't hardcode, test out when you get stuff working
      #if (current_step % 10 == 0):
      #  old_size = model.get_sched("v_step_size")
      #  model.set_sched("v_step_size", old_size * 10)

  save_dir = model.write_checkpoint(sess)
  import IPython; IPython.embed()
