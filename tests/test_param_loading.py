import os
import sys

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.loaders as loaders

def test_param_loading():
    dsc_dir = os.path.join(ROOT_DIR, "DeepSparseCoding")
    params_list = loaders.get_params_list(dsc_dir)
    for params_name in params_list:
        if 'test_' not in params_name:
            params_file = os.path.join(*[dsc_dir, 'params', params_name+'.py'])
            params = loaders.load_params(params_file, key='params')
