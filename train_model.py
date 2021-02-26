import os
import sys
import argparse
import time as ti
from os.path import dirname as up

ROOT_DIR = up(up(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import torch

import DeepSparseCoding.utils.loaders as loaders
import DeepSparseCoding.utils.run_utils as run_utils
import DeepSparseCoding.utils.dataset_utils as dataset_utils

parser = argparse.ArgumentParser()
parser.add_argument('param_file', help='Path to the parameter file')

args = parser.parse_args()
param_file = args.param_file

t0 = ti.time()

# Load params
params = loaders.load_params_file(param_file)

# Load data
train_loader, val_loader, test_loader, data_stats = dataset_utils.load_dataset(params)
for key, value in data_stats.items():
    setattr(params, key, value)

# Load model
model = loaders.load_model(params.model_type)
model.setup(params)
model.to(params.device)
model.log_architecture_details()

# Train model
for epoch in range(1, model.params.num_epochs+1):
    run_utils.train_epoch(epoch, model, train_loader)
    # TODO: Ensemble models might not actually have a classification objective / need validation
    #if(model.params.model_type.lower() in ['mlp', 'ensemble']): # TODO: use to validation set here; test at the end of training
    #    run_utils.test_epoch(epoch, model, test_loader)
    model.logger.log_string(f'Completed epoch {epoch}/{model.params.num_epochs}')
    print(f'Completed epoch {epoch}/{model.params.num_epochs}')

# Final outputs
t1 = ti.time()
tot_time=float(t1-t0)
tot_images = model.params.num_epochs*len(train_loader.dataset)
out_str = f'Training on {tot_images} images is complete. Total time was {tot_time} seconds.\n'
model.logger.log_string(out_str)
print('Training Complete\n')

model.write_checkpoint()
