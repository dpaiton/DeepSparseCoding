import torch.nn as nn

import DeepSparseCoding.utils.loaders as loaders
from DeepSparseCoding.utils.data_processing import flatten_feature_map


class EnsembleModule(nn.Sequential):
    def __init__(self): # do not do Sequential's init
        super(nn.Sequential, self).__init__()

    def setup_ensemble_module(self, params):
        self.params = params
        for subparams in params.ensemble_params:
            submodule = loaders.load_module(subparams.model_type)
            submodule.setup_module(subparams)
            self.add_module(subparams.model_type, submodule)

    def forward(self, x):
        self.layer_list = [x]
        for module in self:
            if module.params.layer_types[0] == 'fc':
                self.layer_list[-1] = flatten_feature_map(self.layer_list[-1])
            self.layer_list.append(module.get_encodings(self.layer_list[-1])) # latent encodings
        return self.layer_list[-1]

    def get_encodings(self, x):
        return self.forward(x)
