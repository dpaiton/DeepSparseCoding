import re
import time
import types
import os
from copy import deepcopy
from collections import OrderedDict
import importlib

import numpy as np
import json as js
import torch


class Logger(object):
    def __init__(self, filename=None, overwrite=True):
        self.filename = filename
        if(filename is None):
            self.log_to_file = False
        else:
            self.log_to_file = True
            if(overwrite):
                self.file_obj = open(filename, 'w', buffering=1)
            else:
                self.file_obj = open(filename, 'r+', buffering=1)
            self.file_obj.seek(0)

    def js_dumpstring(self, obj):
        """Dump json string with special CustomEncoder"""
        return js.dumps(obj, sort_keys=True, indent=2, cls=CustomEncoder)

    def log_string(self, string):
        """Log input string"""
        now = time.localtime(time.time())
        time_str = time.strftime('%m/%d/%y %H:%M:%S', now)
        out_str = '\n' + time_str + ' -- ' + str(string)
        if(self.log_to_file):
            self.file_obj.write(out_str)
        else:
            print(out_str)

    def log_trainable_variables(self, name_list):
        """
        Use logging to write names of trainable variables in model
        Inputs:
            name_list: list containing variable names
        """
        js_str = self.js_dumpstring(name_list)
        self.log_string('<train_vars>'+js_str+'</train_vars>')

    def log_params(self, params):
        """
        Use logging to write model params
        Inputs:
            params: [dict] containing parameters values
        """
        out_params = deepcopy(params)
        if('ensemble_params' in out_params.keys()):
            for sub_idx, sub_params in enumerate(out_params['ensemble_params']):
                #sub_params.set_params()
                for key, value in sub_params.__dict__.items():
                    if(key != 'rand_state'):
                        new_dict_key = f'{sub_idx}_{key}'
                        out_params[new_dict_key] = value
            del out_params['ensemble_params']
        if('rand_state' in out_params.keys()):
            del out_params['rand_state']
        js_str = self.js_dumpstring(out_params)
        self.log_string('<params>'+js_str+'</params>')

    def log_stats(self, stat_dict):
        """Log dictionary of training / testing statistics"""
        js_str = self.js_dumpstring(stat_dict)
        self.log_string('<stats>'+js_str+'</stats>')

    def log_info(self, info_dict):
        """Log input dictionary in <info> tags"""
        js_str = self.js_dumpstring(info_dict)
        self.log_string('<info>'+js_str+'</info>')

    def load_file(self, filename=None):
        """
        Load log file into memory
        Outputs:
            log_text: [str] containing log file text
        """
        if(filename is None):
            self.file_obj.seek(0)
        else:
            self.file_obj = open(filename, 'r', buffering=1)
        text = self.file_obj.read()
        return text

    def read_js(self, tokens, text):
        """
        Read js string encased by tokens and convert to python object
        Outpus:
            output: converted python object
        Inputs:
            tokens: [list] of length 2 with [0] entry indicating start token and [1]
                entry indicating end token
            text: [str] containing text to parse, can be obtained by calling load_file()
        TODO: Verify that js_matches is the same type for both conditionals at the end
            js_matches should be a list at all times. That way when e.g. read_params
            is called the output is a list no matter how many params specifications there are
            in the logfile.
        """
        assert type(tokens) == list, ('Input variable tokens must be a list')
        assert len(tokens) == 2, ('Input variable tokens must be a list of length 2')
        matches = re.findall(re.escape(tokens[0])+r'([\s\S]*?)'+re.escape(tokens[1]), text)
        if(len(matches) > 1):
            js_matches = [js.loads(match) for match in matches]
        else:
            js_matches = [js.loads(matches[0])]
        return js_matches

    def read_params(self, text):
        """
        Read params from text file and return as a params object or list of params objects
        Outpus:
            params: converted python object
        Inputs:
            text: [str] containing text to parse, can be obtained by calling load_file()
        """
        tokens = ['<params>', '</params>']
        params = self.read_js(tokens, text)
        param_list = []
        for param_dict in params:
            param_obj = type('param_obj', (), {})()
            if(param_dict['model_type'] == 'ensemble'):
                param_obj.ensemble_params = []
                ensemble_nums = set()
            for key, value in param_dict.items():
                if(param_dict['model_type'] == 'ensemble'):
                    key_split = key.split('_')
                    if(key_split[0].isdigit()): # ensemble params are prefaced with ensemble index
                        ens_num = int(key_split[0])
                        if(ens_num not in ensemble_nums):
                            ensemble_nums.add(ens_num)
                            param_obj.ensemble_params.append(types.SimpleNamespace())
                        setattr(param_obj.ensemble_params[ens_num], '_'.join(key_split[1:]), value)
                    else: # if it is not a digit then it is a general param
                        setattr(param_obj, key, value)
                else:
                    setattr(param_obj, key, value)

            def optimizer_dict_to_obj(param_obj):
                if(hasattr(param_obj, 'optimizer')): # convert optimizer dict to class
                    optimizer_dict = deepcopy(param_obj.optimizer)
                    param_obj.optimizer = types.SimpleNamespace()
                    for key, value in optimizer_dict.items():
                        setattr(param_obj.optimizer, key, value)

            if(param_obj.model_type == 'ensemble'): # each model in ensembles could have optimizers
                for model_param_obj in param_obj.ensemble_params:
                    optimizer_dict_to_obj(model_param_obj)
            else:
                optimizer_dict_to_obj(param_obj)
            param_list.append(param_obj)
        return param_list

    def read_stats(self, text):
        """
        Generate dictionary of lists that contain stats from log text
        Outpus:
            stats: [dict] containing run statistics
        Inputs:
            text: [str] containing text to parse, can be obtained by calling load_file()
        """
        tokens = ['<stats>', '</stats>']
        js_matches = self.read_js(tokens, text)
        stats = {}
        for js_match in js_matches:
            if(type(js_match) is str):
                js_match = {js_match:js_match}
            for key in js_match.keys():
                if(key in stats):
                    stats[key].append(js_match[key])
                else:
                    stats[key] = [js_match[key]]
        return stats

    def read_architecture(self, text):
        """
        Generate dictionary of lists that contain stats from log text
        Outpus:
            stats: [dict] containing run statistics
        Inputs:
            text: [str] containing text to parse, can be obtained by calling load_file()
        """
        tokens = ['<architecture>', '</architecture>']
        js_match = self.read_js(tokens, text)
        return js_match

    def __del__(self):
        if(self.log_to_file and hasattr(self, 'file_obj')):
            self.file_obj.close()


class CustomEncoder(js.JSONEncoder):
    def default(self, obj):
        if(callable(obj)):
            return None
        elif(isinstance(obj, np.integer)):
            return int(obj)
        elif(isinstance(obj, np.floating)):
            return float(obj)
        elif(isinstance(obj, np.ndarray)):
            return obj.tolist()
        elif(isinstance(obj, torch.device)):
            return obj.type
        elif(isinstance(obj, torch.dtype)):
            return str(obj)
        elif(isinstance(obj, types.SimpleNamespace)):
            return obj.__dict__
        else:
            return super(CustomEncoder, self).default(obj)


def python_module_from_file(py_module_name, file_name):
    assert os.path.isfile(file_name), (f'Error: {file_name} does not exist!')
    spec = importlib.util.spec_from_file_location(py_module_name, file_name)
    py_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(py_module)
    return py_module

def summary_string(model, input_size, batch_size=2, device=torch.device('cuda:0'), dtype=torch.FloatTensor):
    """
    Returns a string that summarizees the model architecture, including the number of parameters
    and layer output sizes

    Code is modified from:
    https://github.com/sksq96/pytorch-summary

    Keyword arguments:
        model [torch module, module subclass, or EnsembleModel] model to summarize
        input_size  [tuple or list of tuples] must not include the batch dimension; if it is a list
            of tuples then the architecture will be computed for each option
        batch_size [positive int] how many images to feed into the model.
            The default of 2 will ensure that batch norm works.
        devie [torch.device] which device to run the test on
        dtype [torch.dtype] for the artificially generated inputs
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = batch_size
            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params
            summary[m_key]['gpu_mem'] = round(torch.cuda.memory_allocated(0)/1024**3, 1)
        if len(list(module.children())) == 0: # only apply hooks at child modules to avoid applying them twice
            hooks.append(module.register_forward_hook(hook))
    x = torch.rand(batch_size, *input_size).type(dtype).to(device=device)
    summary = OrderedDict() # used within hook function to store properties
    hooks = [] # used within hook function to store resgistered hooks
    model.apply(register_hook) # recursively apply register_hook function to model and all children
    model(x) # make a forward pass
    for h in hooks:
        h.remove() # remove the hooks so they are not used at run time
    summary_str = '----------------------------------------------------------------\n'
    line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
    summary_str += line_new + '\n'
    summary_str += '================================================================\n'
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = '{:>20}  {:>25} {:>15}'.format(
            layer,
            str(summary[layer]['output_shape']),
            '{0:,}'.format(summary[layer]['nb_params']),
        ) # input_shape, output_shape, trainable, nb_params
        total_params += summary[layer]['nb_params']
        total_output += np.prod(summary[layer]['output_shape'])
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        summary_str += line_new + '\n'
    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size)) * batch_size * 4. / (1024 ** 2.)
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    summary_str += '================================================================\n'
    summary_str += f'Total params: {total_params}\n'
    summary_str += f'Trainable params: {trainable_params}\n'
    param_diff = total_params - trainable_params
    summary_str += f'Non-trainable params: {param_diff}\n'
    summary_str += '----------------------------------------------------------------\n'
    summary_str += f'Input size (MB): {total_input_size:0.2f}\n'
    summary_str += f'Forward/backward pass size (MB): {total_output_size:0.2f}\n'
    summary_str += f'Params size (MB): {total_params_size:0.2f}\n'
    summary_str += f'Estimated total size (MB): {total_size:0.2f}\n'
    ## TODO: Update pytorch for this to work
    #device_memory = torch.cuda.memory_summary(device, abbreviated=True)
    #summary_str += f'Device memory allocated with batch of inputs (GB): {device_memory}\n'
    summary_str += '----------------------------------------------------------------\n'
    return summary_str, (total_params, trainable_params)
