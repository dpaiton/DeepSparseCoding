import os
import pprint

import numpy as np
import torch

from DeepSparseCoding.utils.file_utils import Logger
import DeepSparseCoding.utils.loaders as loaders


class BaseModel(object):
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

    def load_params(self, params):
        """
        Calculates a few extra parameters
        Sets parameters as member variable
        """
        if not hasattr(params, 'model_out_dir'):
            params.model_out_dir = os.path.join(params.out_dir, params.model_name)
        params.cp_save_dir = os.path.join(params.model_out_dir, 'checkpoints')
        params.log_dir = os.path.join(params.model_out_dir, 'logfiles')
        params.save_dir = os.path.join(params.model_out_dir, 'savefiles')
        params.disp_dir = os.path.join(params.model_out_dir, 'vis')
        params.batches_per_epoch = params.epoch_size / params.batch_size
        params.num_batches = params.num_epochs * params.batches_per_epoch
        if not  hasattr(params, "cp_latest_filename"):
            params.cp_latest_filename = os.path.join(params.cp_save_dir,
                f'{params.model_name}_latest_checkpoint_v{params.version}.pt')
        self.params = params

    def check_params(self):
        """
        Check parameters with assertions
        """
        assert self.params.num_pixels == int(np.prod(self.params.data_shape))
        if self.params.device is torch.device('cpu'):
            print('WARNING: Model is running on the CPU')

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
                log_filename = os.path.join(self.params.log_dir,
                    self.params.model_name+'_v'+self.params.version+'.log')
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

    def get_train_stats(self, batch_step=None):
        """
        Get default statistics about current training run

        Keyword arguments:
            batch_step: [int] current batch iteration. The default assumes that training has finished.
        """
        if batch_step is None:
            batch_step = self.params.num_batches
        epoch = batch_step / self.params.batches_per_epoch
        stat_dict = {
            'epoch':int(epoch),
            'batch_step':batch_step,
            'train_progress':np.round(batch_step/self.params.num_batches, 3),
        }
        return stat_dict

    def write_checkpoint(self, batch_step=None):
        """
        Write checkpoints

        Keyword arguments:
            batch_step: [int] current batch iteration. The default assumes that training has finished.
        """
        output_dict = {
            'model_state_dict': self.state_dict(),
        }
        if(self.params.model_type.lower() == 'ensemble'):
            for module in self:
                module_state_dict_name = module.params.submodule_name+'_optimizer_state_dict'
                output_dict[module_state_dict_name] = module.optimizer.state_dict(),
        else:
            module_state_dict_name = 'optimizer_state_dict'
            output_dict[module_state_dict_name] = self.optimizer.state_dict(),
        training_stats = self.get_train_stats(batch_step)
        output_dict.update(training_stats)
        torch.save(output_dict, self.params.cp_latest_filename)
        self.log_info('Full model saved in file %s'%self.params.cp_latest_filename)

    def get_checkpoint_from_log(self, logfile):
        model_params = loaders.load_params_from_log(logfile)
        checkpoint = torch.load(model_params.cp_latest_filename)
        return checkpoint

    def load_checkpoint(self, cp_file=None, load_optimizer=False):
        """
        Load checkpoint
        Keyword arguments:
          model_dir: [str] specifying the path to the checkpoint
        """
        if cp_file is None:
            cp_file = self.params.cp_latest_filename
        checkpoint = torch.load(cp_file)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.load_state_dict(checkpoint['model_state_dict'])
        _ = checkpoint.pop('optimizer_state_dict', None)
        _ = checkpoint.pop('model_state_dict', None)
        training_status = pprint.pformat(checkpoint, compact=True)#, sort_dicts=True #TODO: Python 3.8 adds the sort_dicts parameter
        out_str = f'Loaded checkpoint from {cp_file} with the following stats:\n{training_status}'
        return out_str

    def get_optimizer(self, optimizer_params, trainable_variables):
        optimizer_name = optimizer_params.optimizer.name
        if(optimizer_name == 'sgd'):
            optimizer = torch.optim.SGD(
                trainable_variables,
                lr=optimizer_params.weight_lr,
                weight_decay=optimizer_params.weight_decay)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                trainable_variables,
                lr=optimizer_params.weight_lr,
                weight_decay=optimizer_params.weight_decay)
        else:
            assert False, (f'optimizer name must be "sgd" or "adam", not {optimizer_name}')
        return optimizer

    def setup_optimizer(self):
        self.optimizer = self.get_optimizer(
                optimizer_params=self.params,
                trainable_variables=self.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.params.optimizer.milestones,
            gamma=self.params.optimizer.lr_decay_rate)

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
          For example: logging.info('<stats>'+self.js_dumpstring(output_dictionary)+'</stats>')
        """
        update_dict = self.generate_update_dict(input_data, input_labels, batch_step)
        js_str = self.js_dumpstring(update_dict)
        self.log_info('<stats>'+js_str+'</stats>')

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0, update_dict=None):
        """
        Generates a dictionary to be logged in the print_update function
        """
        if update_dict is None:
            update_dict = self.get_train_stats(batch_step)
        for param_name, param_var in self.named_parameters():
            grad = param_var.grad
            update_dict[param_name+'_grad_max_mean_min'] = [
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
