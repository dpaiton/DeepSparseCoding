import re
import time
import types
import os
from copy import deepcopy
import importlib

import numpy as np
import json as js
import torch


class Logger(object):
    def __init__(self, filename=None, overwrite=True):
        self.filename = filename
        if filename is None:
            self.log_to_file = False
        else:
            self.log_to_file = True
            if overwrite:
                self.file_obj = open(filename, "w", buffering=1)
            else:
                self.file_obj = open(filename, "r+", buffering=1)
            self.file_obj.seek(0)

    def js_dumpstring(self, obj):
        """Dump json string with special CustomEncoder"""
        return js.dumps(obj, sort_keys=True, indent=2, cls=CustomEncoder)

    def log_trainable_variables(self, name_list):
        """
        Use logging to write names of trainable variables in model
        Inputs:
            name_list: list containing variable names
        """
        js_str = self.js_dumpstring(name_list)
        self.log_info("<train_vars>"+js_str+"</train_vars>")

    def log_params(self, params):
        """
        Use logging to write model params
        Inputs:
            params: [dict] containing parameters values
        """
        out_params = deepcopy(params)
        if "ensemble_params" in out_params.keys():
            for sub_idx, sub_params in enumerate(out_params["ensemble_params"]):
                sub_params.set_params()
                for key, value in sub_params.__dict__.items():
                    if key != "rand_state":
                        out_params[str(sub_idx)+"_"+key] = value
            del out_params["ensemble_params"]
        if "rand_state" in out_params.keys():
            del out_params["rand_state"]
        js_str = self.js_dumpstring(out_params)
        self.log_info("<params>"+js_str+"</params>")

    def log_info(self, string):
        """Log input string"""
        now = time.localtime(time.time())
        time_str = time.strftime("%m/%d/%y %H:%M:%S", now)
        out_str = "\n" + time_str + " -- " + str(string)
        if self.log_to_file:
            self.file_obj.write(out_str)
        else:
            print(out_str)

    def load_file(self, filename=None):
        """
        Load log file into memory
        Outputs:
            log_text: [str] containing log file text
        """
        if filename is None:
            self.file_obj.seek(0)
        else:
            self.file_obj = open(filename, "r", buffering=1)
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
            I believe js_matches should be a list at all times. That way when e.g. read_params
            is called the output is a list no matter how many params specifications there are
            in the logfile.
        """
        assert type(tokens) == list, ("Input variable tokens must be a list")
        assert len(tokens) == 2, ("Input variable tokens must be a list of length 2")
        matches = re.findall(re.escape(tokens[0])+"([\s\S]*?)"+re.escape(tokens[1]), text)
        if len(matches) > 1:
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
        tokens = ["<params>", "</params>"]
        params = self.read_js(tokens, text)
        param_list = []
        for param_dict in params:
            param_obj = type("param_obj", (), {})()
            for key, value in param_dict.items():
                  setattr(param_obj, key, value)
            if hasattr(param_obj, "optimizer"): # convert optimizer dict to class
                optimizer_dict = deepcopy(param_obj.optimizer)
                param_obj.optimizer = types.SimpleNamespace()
                for key, value in optimizer_dict.items():
                    setattr(param_obj.optimizer, key, value)
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
        tokens = ["<stats>", "</stats>"]
        js_matches = self.read_js(tokens, text)
        stats = {}
        for js_match in js_matches:
            if type(js_match) is str:
                js_match = {js_match:js_match}
            for key in js_match.keys():
                if key in stats:
                    stats[key].append(js_match[key])
                else:
                    stats[key] = [js_match[key]]
        return stats

    def __del__(self):
        if self.log_to_file and hasattr(self, "file_obj"):
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


def module_from_file(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
