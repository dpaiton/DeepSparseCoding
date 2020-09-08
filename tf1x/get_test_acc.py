import os
import sys
root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)
projects_dir = root_path+'/Projects/'

import numpy as np

import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap


def load_analysis(params):
  analyzer = ap.get_analyzer(params.model_type)
  analyzer.setup(params)
  analyzer.model_name = params.model_name
  return analyzer


mnist_lca_768_2layer = 'slp_lca_768_latent_mnist'#'slp_lca_768_latent_75_steps_mnist'
mnist_lca_768_3layer = 'mlp_lca_768_latent_75_steps_mnist'
mnist_lca_1568_2layer = 'slp_lca_1568_latent_mnist'#'slp_lca_1568_latent_75_steps_mnist'
mnist_lca_1568_3layer = 'mlp_lca_1568_latent_75_steps_mnist'
mnist_mlp_768_2layer = 'mlp_768_mnist'#'mlp_cosyne_mnist'
mnist_mlp_768_3layer = 'mlp_3layer_cosyne_mnist'
mnist_mlp_1568_2layer = 'mlp_1568_mnist'
mnist_mlp_1568_3layer = 'mlp_1568_3layer_mnist'

class mlp_lca_768n_2l(object):
  def __init__(self):
    self.model_type = 'mlp_lca'
    self.model_name = mnist_lca_768_2layer
    self.display_name = 'LCA;768N;2L'


class mlp_lca_768n_3l(object):
  def __init__(self):
    self.model_type = 'mlp_lca'
    self.model_name = mnist_lca_768_3layer
    self.display_name = 'LCA;768N;3L'


class mlp_lca_1568n_2l(object):
  def __init__(self):
    self.model_type = 'mlp_lca'
    self.model_name = mnist_lca_1568_2layer
    self.display_name = 'LCA;1568N;2L'


class mlp_lca_1568n_3l(object):
  def __init__(self):
    self.model_type = 'mlp_lca'
    self.model_name = mnist_lca_1568_3layer
    self.display_name = 'LCA;1568N;3L'


class mlp_768n_2l(object):
  def __init__(self):
    self.model_type = 'mlp'
    self.model_name = mnist_mlp_768_2layer
    self.display_name = 'MLP;768N;2L'


class mlp_768n_3l(object):
  def __init__(self):
    self.model_type = 'mlp'
    self.model_name = mnist_mlp_768_3layer
    self.display_name = 'MLP;768N;3L'


class mlp_1568n_2l(object):
  def __init__(self):
    self.model_type = 'mlp'
    self.model_name = mnist_mlp_1568_2layer
    self.display_name = 'MLP;1568N;2L'


class mlp_1568n_3l(object):
  def __init__(self):
    self.model_type = 'mlp'
    self.model_name = mnist_mlp_1568_3layer
    self.display_name = 'MLP;1568N;3L'


params_list = [
    mlp_lca_768n_2l(),
    mlp_lca_768n_3l(),
    mlp_lca_1568n_2l(),
    mlp_lca_1568n_3l(),
    mlp_768n_2l(),
    mlp_768n_3l(),
    mlp_1568n_2l(),
    mlp_1568n_3l()
]

analyzers = []
display_names = []
for params in params_list:
    params.version = '0.0'
    params.model_dir = projects_dir+params.model_name
    params.save_info = 'analysis_test_kurakin_targeted'
    params.overwrite_analysis_log = False
    analyzer = load_analysis(params)
    analyzer.model_params.batch_size = 500
    analyzers.append(analyzer)
    display_names.append(params.display_name)

# all models should have the same preprocessing
data = ds.get_data(analyzers[0].model_params)
data = analyzers[0].model.preprocess_dataset(data, analyzers[0].model_params)
data = analyzers[0].model.reshape_dataset(data, analyzers[0].model_params)

accuracies = []
for analyzer in analyzers:
    analyzer.model_params.data_shape = list(data['test'].shape[1:])
    analyzer.model.setup(analyzer.model_params)
    test_images = data["test"].images
    test_labels = data["test"].labels

    softmaxes = np.squeeze(analyzer.compute_activations(
        test_images,
        batch_size=analyzer.model_params.batch_size,
        activation_operation=analyzer.model.get_label_est
    ))
    test_accuracy = np.mean(np.argmax(test_labels, -1) == np.argmax(softmaxes, -1))
    accuracies.append(test_accuracy)

for display_name, accuracy in zip(display_names, accuracies):
    print(f'model {display_name} had test accuracy = {accuracy:.5f}')
