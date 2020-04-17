import os
import sys
import argparse
import time as ti

parent_path = os.path.dirname(os.getcwd())
if parent_path not in sys.path: sys.path.append(parent_path)

import torch

import PyTorchDisentanglement.params.param_loader as pl
import PyTorchDisentanglement.models.model_loader as ml
import PyTorchDisentanglement.utils.run_utils as run_utils
import PyTorchDisentanglement.utils.dataset_utils as dataset_utils

parser = argparse.ArgumentParser()
parser.add_argument("param_file", help="Path to the parameter file")

args = parser.parse_args()
param_file = args.param_file

t0 = ti.time()

# Load params
params = pl.load_param_file(param_file)

# Load data
train_loader, val_loader, test_loader, params = dataset_utils.load_dataset(params)

# Load model
model = ml.load_model(params.model_type)
model.setup(params)
model.to(params.device)

# Train model
for epoch in range(1, model.params.num_epochs+1):
   run_utils.train_epoch(epoch, model, train_loader)
   if(model.params.model_type.lower() in ["mlp", "ensemble"]):
       run_utils.test_epoch(epoch, model, test_loader)
   print("Completed epoch %g/%g"%(epoch, model.params.num_epochs))

t1 = ti.time()
tot_time=float(t1-t0)
out_str = (
  "Training on "+str(model.params.num_epochs*len(train_loader.dataset))
  +" Images is Complete. Total time was "+str(tot_time)+" seconds.\n")
model.log_info(out_str)
print("Training Complete\n")

# Checkpoint model
PATH = model.params.cp_save_dir
if not os.path.exists(PATH):
    os.makedirs(PATH)
SAVEFILE = PATH + "trained_model.pt"
torch.save(model.state_dict(), SAVEFILE)
