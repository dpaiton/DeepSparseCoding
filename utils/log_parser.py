import re
import json
import numpy as np

"""
Load log file into memory
Outputs:
  log_text: [str] containing log file text
Inputs:
  log_file: [str] containing the log filename
"""
def load_file(log_file):
  with open(log_file, "r") as f:
    log_text = f.read()
  return log_text

"""
Extract values from input text using preceding token
Outputs:
  output_val: [...] output value casted to appropriate type
Inputs:
  token: [str] prefix token for searching for values
  log_text: [str] text to find values in
"""
def extract_vals(token, log_text):
  print(token)
  #if 'decay_rate' in token:
  #  import IPython; IPython.embed()
  val_match = re.search(token+" = ([^\t\n\r\f\v<]+)", log_text)
  val = log_text[val_match.start():val_match.end()].split(" = ")[1].strip()
  type_match  = re.search((token+" = "+re.escape(val)+" (\S+ \S+)"), log_text)
  log_text_segment = log_text[type_match.start():type_match.end()]
  val_type = log_text_segment.split("<")[1].split("'")[1]
  if val_type == "list":
    val_str = log_text[val_match.start():val_match.end()].split(" = ")[1]
    if 'True' or 'False' in val_str:
      val_str = val_str.lower()
    output_val = json.loads(val_str.replace("'",'"'))
  else:
    output_val = eval(val_type)(val)
  return output_val

"""
Generate dictionary of model parameters
Outpus:
  params: [dict] containing model parameters
Inputs:
  log_text: [str] containing log text, can be obtained by calling load_file()
"""
def read_params(log_text):
  params = {}
  keys = re.findall("param: (\S+) =", log_text)
  for key in keys:
    params[key] = extract_vals("param: "+key, log_text)
  return params

"""
Generate lis of dictionariies for the model schedule
Outpus:
  params: [list] containing model schedule dictionary
Inputs:
  log_text: [str] containing log text, can be obtained by calling load_file()
"""
def read_schedule(log_text):
  schedule = []
  sched_indices = re.findall("sched_(\d+)", log_text)
  for sched_idx_str in sorted(set(sched_indices)):
    sched_idx = int(sched_idx_str)
    schedule.append({})
    keys = re.findall("sched_"+sched_idx_str+": (\S+) =", log_text)
    for key in keys:
      schedule[sched_idx][key] = extract_vals("sched_"+sched_idx_str+": "+key,
        log_text)
  return schedule
