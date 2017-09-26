import re
import json as js

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
Read js string encased by tokens and convert to python object
Outpus:
  output: converted python object
Inputs:
  tokens: [list] of length 2 with [0] entry indicating start token and [1]
    entry indicating end token
  text: [str] containing text to parse, can be obtained by calling load_file()
"""
def read_js(tokens, text):
  assert type(tokens) == list, ("Input variable tokens must be a list")
  assert len(tokens) == 2, ("Input variable tokens must be a list of length 2")
  matches = re.findall(re.escape(tokens[0])+"([\s\S]*?)"+re.escape(tokens[1]),
    text)
  if len(matches) > 1:
    js_matches = [js.loads(match) for match in matches]
  else:
    js_matches = js.loads(matches[0])
  return js_matches

"""
Read params from text file and return as a dictionary
Outpus:
  params: converted python object
Inputs:
  text: [str] containing text to parse, can be obtained by calling load_file()
"""
def read_params(text):
  tokens = ["<params>", "</params>"]
  return read_js(tokens, text)

"""
Read schedule from text file and return as a list of dictionaries
Outpus:
  schedule: converted python object
Inputs:
  text: [str] containing text to parse, can be obtained by calling load_file()
"""
def read_schedule(text):
  tokens = ["<schedule>", "</schedule>"]
  return read_js(tokens, text)

"""
Generate dictionary of lists that contain stats from log text
Outpus:
  stats: [dict] containing run statistics
Inputs:
  text: [str] containing text to parse, can be obtained by calling load_file()
"""
def read_stats(text):
  tokens = ["<stats>", "</stats>"]
  js_matches = read_js(tokens, text)
  stats = {}
  for js_match in js_matches:
    for key in js_match.keys():
      if key in stats:
        stats[key].append(js_match[key])
      else:
        stats[key] = [js_match[key]]
  return stats
