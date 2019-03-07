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

## import model and process data
model = mp.get_model(model_type)
data = model.preprocess_dataset(data, params)
data = model.reshape_dataset(data, params)
params.data_shape = list(data["train"].shape[1:])
model.setup(params)


# setup tensorflow model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True



