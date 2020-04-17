import utils.file_utils as file_utils
#import data.get_data_shape

def load_param_file(file_name):
    params_module = file_utils.module_from_file("params", file_name)
    params = params_module.params()
    #params.data_shape = data.get_data_shape.get_data_shape(params.dataset)
    return params


if __name__ == "__main__":
    # Load params from file, parse it, print it out
    print("TODO.")
