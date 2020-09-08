import os
import sys

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.file_utils as file_utils

def get_dir_list(target_dir, target_string):
    dir_list = [filename.split('.')[0]
        for filename in os.listdir(target_dir)
        if target_string in filename]
    return dir_list

def get_model_list(root_search_dir):
    return get_dir_list(os.path.join(root_search_dir, "models"), "_model")


def get_params_list(root_search_dir):
    return get_dir_list(os.path.join(root_search_dir, "params"), "_params")


def get_module_list(root_search_dir):
    return get_dir_list(os.path.join(root_search_dir, "modules"), "_module")


def get_all_lists(root_search_dir):
    model_list = get_model_list(root_search_dir)
    params_list = get_params_list(root_search_dir)
    module_list = get_module_list(root_search_dir)
    return (params_list, model_list, module_list)


def load_model_class(model_type):
    dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
    if(model_type.lower() == 'base'):
        py_module_name = 'BaseModel'
        file_name = os.path.join(*[dsc_dir, 'models', 'base_model.py'])
    elif(model_type.lower() == 'mlp'):
        py_module_name = 'MlpModel'
        file_name = os.path.join(*[dsc_dir, 'models', 'mlp_model.py'])
    elif(model_type.lower() == 'lca'):
        py_module_name = 'LcaModel'
        file_name = os.path.join(*[dsc_dir, 'models', 'lca_model.py'])
    elif(model_type.lower() == 'conv_lca'):
        py_module_name = 'ConvLcaModel'
        file_name = os.path.join(*[dsc_dir, 'models', 'conv_lca_model.py'])
    elif(model_type.lower() == 'ensemble'):
        py_module_name = 'EnsembleModel'
        file_name = os.path.join(*[dsc_dir, 'models', 'ensemble_model.py'])
    else:
        accepted_names = [''.join(name.split('_')[:-1]) for name in get_module_list(dsc_dir)]
        assert False, (
            'Acceptible model_types are %s, not %s'%(','.join(accepted_names), model_type))
    py_module = file_utils.python_module_from_file(py_module_name, file_name)
    py_module_class = getattr(py_module, py_module_name)
    return py_module_class


def load_model(model_type):
    return load_model_class(model_type)()


def load_module(module_type):
    dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
    if(module_type.lower() == 'mlp'):
        py_module_name = 'MlpModule'
        file_name = os.path.join(*[dsc_dir, 'modules', 'mlp_module.py'])
    elif(module_type.lower() == 'lca'):
        py_module_name = 'LcaModule'
        file_name = os.path.join(*[dsc_dir, 'modules', 'lca_module.py'])
    elif(module_type.lower() == 'conv_lca'):
        py_module_name = 'ConvLcaModule'
        file_name = os.path.join(*[dsc_dir, 'modules', 'conv_lca_module.py'])
    elif(module_type.lower() == 'ensemble'):
        py_module_name = 'EnsembleModule'
        file_name = os.path.join(*[dsc_dir, 'modules', 'ensemble_module.py'])
    else:
        accepted_names = [''.join(name.split('_')[:-1]) for name in get_module_list(dsc_dir)]
        assert False, (
            'Acceptible model_types are %s, not %s'%(','.join(accepted_names), module_type))
    py_module = file_utils.python_module_from_file(py_module_name, file_name)
    py_module_class = getattr(py_module, py_module_name)
    return py_module_class()


def load_params(file_name, key='params'):
    params_module = file_utils.python_module_from_file(key, file_name)
    params = getattr(params_module, key)()
    return params


if __name__ == '__main__':
    dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
    out_str = '\n\n'.join([
        ', '.join(out_list)
        for out_list in list(get_all_lists(dsc_dir))])
    print('\n'+out_str+'\n')
