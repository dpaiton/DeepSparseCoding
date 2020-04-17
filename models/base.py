import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PyTorchDisentanglement.utils.file_utils import Logger

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.params_loaded = False

    def setup(self, params, logger=None):
        """
        Setup required model components
        #TODO: log system info, including git commit hash
        """
        self.load_params(params)
        self.check_params()
        self.make_dirs()
        if logger is None:
            self.init_logging()
            self.log_params()
        else:
            self.logger = logger
        self.setup_model()
        self.setup_optimizer()

    def load_params(self, params):
        """
        Calculates a few extra parameters
        Sets parameters as member variable
        """
        params.cp_latest_filename = "latest_checkpoint_v"+params.version
        if not hasattr(params, "model_out_dir"):
            params.model_out_dir = params.out_dir + params.model_name
        params.cp_save_dir = params.model_out_dir + "/checkpoints/"
        params.log_dir = params.model_out_dir + "/logfiles/"
        params.save_dir = params.model_out_dir + "/savefiles/"
        params.disp_dir = params.model_out_dir + "/vis/"
        params.batches_per_epoch = params.epoch_size / params.batch_size
        params.num_batches = params.num_epochs * params.batches_per_epoch
        self.params = params
        self.params_loaded = True

    def check_params(self):
        """
        Check parameters with assertions
        """
        assert self.params.num_pixels == int(np.prod(self.params.data_shape))

    def get_param(self, param_name):
        """
        Get param value from model
          This is equivalent to self.param_name, except that it will return None if
          the param does not exist.
        """
        if hasattr(self, param_name):
            return getattr(self, param_name)
        else:
            return None

    def make_dirs(self):
        """Make output directories"""
        if not os.path.exists(self.params.model_out_dir):
            os.makedirs(self.params.model_out_dir)
        if not os.path.exists(self.params.log_dir):
            os.makedirs(self.params.log_dir)
        if not os.path.exists(self.params.cp_save_dir):
            os.makedirs(self.params.cp_save_dir)
        if not os.path.exists(self.params.save_dir):
            os.makedirs(self.params.save_dir)
        if not os.path.exists(self.params.disp_dir):
            os.makedirs(self.params.disp_dir)

    def init_logging(self, log_filename=None):
        if self.params.log_to_file:
            if log_filename is None:
                log_filename = self.params.log_dir+self.params.model_name+"_v"+self.params.version+".log"
                self.logger = Logger(filename=log_filename, overwrite=True)
        else:
            self.logger = Logger(filename=None)

    def js_dumpstring(self, obj):
        """Dump json string with special NumpyEncoder"""
        return self.logger.js_dumpstring(obj)

    def log_params(self, params=None):
        """Use logging to write model params"""
        if params is not None:
            dump_obj = params.__dict__
        else:
            dump_obj = self.params.__dict__
        self.logger.log_params(dump_obj)

    def log_info(self, string):
        """Log input string"""
        self.logger.log_info(string)

    def write_checkpoint(self, session):
        """Write checkpoints"""
        base_save_path = self.params.cp_save_dir+self.params.model_name+"_v"+self.params.version
        full_save_path = base_save_path+self.params.cp_latest_filename
        torch.save(self.state_dict(), full_save_path)
        self.logger.log_info("Full model saved in file %s"%full_save_path)
        return base_save_path

    def load_checkpoint(self, model_dir):
        """
        Load checkpoint model into session.
        Inputs:
          model_dir: String specifying the path to the checkpoint
        """
        assert self.params.cp_load == True, ("cp_load must be set to true to load a checkpoint")
        cp_file = model_dir+self.params.cp_latest_filename
        return torch.load(cp_file)

    def setup_model(self):
        raise NotImplementedError

    def get_optimizer(self, optimizer_params, trainable_variables):
        optimizer_name = optimizer_params.optimizer.name
        if(optimizer_name == "sgd"):
            optimizer = torch.optim.SGD(
                trainable_variables,
                lr=optimizer_params.weight_lr,
                weight_decay=optimizer_params.weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                trainable_variables,
                lr=optimizer_params.weight_lr,
                weight_decay=optimizer_params.weight_decay)
        else:
            assert False, ("optimizer name must be 'sgd' or 'adam', not %s"%(optimizer_name))
        return optimizer

    def setup_optimizer(self):
        self.optimizer = self.get_optimizer(
                optimizer_params=self.params,
                trainable_variables=self.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.params.optimizer.milestones,
            gamma=self.params.optimizer.lr_decay_rate)

    def get_encodings(self):
        raise NotImplementedError

    def print_update(self, input_data, input_labels=None, batch_step=0):
        """
        Log train progress information
        Inputs:
          input_data: data object containing the current image batch
          input_labels: data object containing the current label batch
          batch_step: current batch number within the schedule
        NOTE: For the analysis code to parse update statistics, the self.js_dumpstring() call
          must receive a dict object. Additionally, the self.js_dumpstring() output must be
          logged with <stats> </stats> tags.
          For example: logging.info("<stats>"+self.js_dumpstring(output_dictionary)+"</stats>")
        """
        update_dict = self.generate_update_dict(input_data, input_labels, batch_step)
        js_str = self.js_dumpstring(update_dict)
        self.log_info("<stats>"+js_str+"</stats>")

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        """
        Generates a dictionary to be logged in the print_update function
        """
        update_dict = dict()
        for param_name, param_var in self.named_parameters():
            grad = param_var.grad
            update_dict[param_name+"_grad_max_mean_min"] = [
                grad.max().item(), grad.mean().item(), grad.min().item()]
        return update_dict

    def generate_plots(self, input_data, input_labels=None):
        """
        Plot weights, reconstruction, gradients, etc
        Inputs:
          input_data: data object containing the current image batch
          input_labels: data object containing the current label batch
        """
        pass
