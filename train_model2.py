import tensorflow as tf
import numpy as np 
import matplotlib
matplotlib.use("Agg")

import params.param_picker as pp
import models.model_picker as mp 
import data.data_selector as ds

import time

# specify model
model_type = "ica_subspace"
data_type = "vanHateren"


## import params
params = pp.get_params(model_type)
params.data_type = data_type
params.data_dir = "/home/ryanchan/datasets"


## import data
data = ds.get_data(params)

## import model and  rocess data
model = mp.get_model(model_type)
data = model.preprocess_dataset(data, params)
data = model.reshape_dataset(data, params)
params.data_shape = list(data["train"].shape[1:])
model.setup(params)


# setup tensorflow model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# train model 
with tf.Session(config=config, graph=model.graph) as sess:
#    sess.run(model.inti_op)
    
    sess.run(model.init_op)
    for sch_idx, sch in enumerate(params.schedule):
        for b_step in range(sch["num_batches"]):
            data_batch = data["train"].next_batch(params.batch_size)
            input_data = data_batch[0]
            input_labels = data_batch[1]
            #model.generate_update_dict(input_data, input_labels)

            #print("input_data")
            #print(input_data)

            feed_dict = model.get_feed_dict(input_data, input_labels)
            sess.run(model.apply_grads[sch_idx][0], feed_dict)
            #print("model w synth")
            #print(sess.run(model.w_synth))
            
            if b_step % 10 == 0:
                print("step: {}".format(b_step))
                model.print_update(input_data, input_labels, b_step+1)
                model.generate_plots(input_data, input_labels)
