import os

import DeepSparseCoding.utils.file_utils as file_utils

def load_model_class(model_type, root_dir):
    if(model_type.lower() == 'mlp'):
        py_module_name = 'MlpModel'
        file_name = os.path.join(root_dir, 'models/mlp_model.py')
    elif(model_type.lower() == 'lca'):
        py_module_name = 'LcaModel'
        file_name = os.path.join(root_dir, 'models/lca_model.py')
    elif(model_type.lower() == 'ensemble'):
        py_module_name = 'EnsembleModel'
        file_name = os.path.join(root_dir, 'models/ensemble_model.py')
    else:
        accepted_names = ['mlp', 'lca', 'ensemble']
        assert False, (
            'Acceptible model_types are %s, not %s'%(','.join(accepted_names), model_type))
    py_module = file_utils.python_module_from_file(py_module_name, file_name)
    py_module_class = getattr(py_module, py_module_name)
    return py_module_class

def load_model(model_type, root_dir):
    return load_model_class(model_type, root_dir)()

def load_module(module_type, root_dir):
    if(module_type.lower() == 'mlp'):
        py_module_name = 'MlpModule'
        file_name = os.path.join(root_dir, 'modules/mlp_module.py')
    elif(module_type.lower() == 'lca'):
        py_module_name = 'LcaModule'
        file_name = os.path.join(root_dir, 'modules/lca_module.py')
    elif(module_type.lower() == 'ensemble'):
        py_module_name = 'EnsembleModule'
        file_name = os.path.join(root_dir, 'modules/ensemble_module.py')
    else:
        accepted_names = ['mlp', 'lca', 'ensemble']
        assert False, (
            'Acceptible model_types are %s, not %s'%(','.join(accepted_names), module_type))
    py_module = file_utils.python_module_from_file(py_module_name, file_name)
    py_module_class = getattr(py_module, py_module_name)
    return py_module_class()

def load_params(file_name):
    params_module = file_utils.python_module_from_file('params', file_name)
    params = params_module.params()
    return params

if __name__ == '__main__':
    # print list of models & modules & params
    print('TODO.')
