import torch.nn as nn

import DeepSparseCoding.utils.loaders as loaders


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
            self.layer_list.append(module.get_encodings(self.layer_list[-1])) # latent encodings
        return self.layer_list[-1]

    def get_encodings(self, x):
        return self.forward(x)
