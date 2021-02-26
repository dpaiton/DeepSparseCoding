import torch.nn as nn

import DeepSparseCoding.utils.loaders as loaders


class EnsembleModule(nn.Sequential):
    def setup_ensemble_module(self, params):
        self.params = params
        for subparams in params.ensemble_params:
            submodule = loaders.load_module(subparams.model_type)
            submodule.setup_module(subparams)
            self.add_module(subparams.layer_name, submodule)

    def forward(self, x):
        for module in self:
            if module.params.layer_types[0] == 'fc':
                x = x.view(x.size(0), -1) #flat
            x = module(x)
        return x

    def get_encodings(self, x):
        return self.forward(x)
