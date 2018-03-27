import re
import time
import numpy as np
import json as js

class Logger(object):
  def __init__(self, filename=None):
    if filename is None:
      self.log_to_file = False
    else:
      self.log_to_file = True
      self.file_obj = open(filename, "a+", buffering=1)
      self.file_obj.seek(0)

  def js_dumpstring(self, obj):
    """Dump json string with special NumpyEncoder"""
    return js.dumps(obj, sort_keys=True, indent=2, cls=NumpyEncoder)

  def log_params(self, params):
    """
    Use logging to write model params
    Inputs:
      params: [dict] containing parameters values
    """
    if "rand_state" in params.keys():
      del params["rand_state"]
    js_str = self.js_dumpstring(params)
    self.log_info("<params>"+js_str+"</params>")

  def log_schedule(self, sched):
    """Use logging to write current schedule"""
    js_str = self.js_dumpstring(sched)
    self.log_info("<schedule>"+js_str+"</schedule>")

  def log_info(self, string):
    """Log input string"""
    now = time.localtime(time.time())
    time_str = time.strftime("%m/%d/%y %H:%M:%S", now)
    out_str = "\n" + time_str + " -- " + str(string)
    if self.log_to_file:
      self.file_obj.write(out_str)
    else:
      print(out_str)

  def load_file(self):
    """
    Load log file into memory
    Outputs:
      log_text: [str] containing log file text
    TODO: Make text a member variable, other load functions use member variable
      This would then become an internal function without a return
      Downside is you keep the text in memory even if you're not using it...
      Current setup allows for the text to be dropped once e.g. params are read
    """
    self.file_obj.seek(0)
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
    """
    assert type(tokens) == list, ("Input variable tokens must be a list")
    assert len(tokens) == 2, ("Input variable tokens must be a list of length 2")
    matches = re.findall(re.escape(tokens[0])+"([\s\S]*?)"+re.escape(tokens[1]), text)
    if len(matches) > 1:
      js_matches = [js.loads(match) for match in matches]
    else:
      js_matches = js.loads(matches[0])
    return js_matches

  def read_params(self, text):
    """
    Read params from text file and return as a dictionary
    Outpus:
      params: converted python object
    Inputs:
      text: [str] containing text to parse, can be obtained by calling load_file()
    """
    tokens = ["<params>", "</params>"]
    return self.read_js(tokens, text)

  def read_schedule(self, text):
    """
    Read schedule from text file and return as a list of dictionaries
    Outpus:
      schedule: converted python object
    Inputs:
      text: [str] containing text to parse, can be obtained by calling load_file()
    """
    tokens = ["<schedule>", "</schedule>"]
    return self.read_js(tokens, text)

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
      for key in js_match.keys():
        if key in stats:
          stats[key].append(js_match[key])
        else:
          stats[key] = [js_match[key]]
    return stats

  def __del__(self):
    if self.log_to_file:
      self.file_obj.close()

class NumpyEncoder(js.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(NumpyEncoder, self).default(obj)
