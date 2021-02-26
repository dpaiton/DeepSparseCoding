import pprint

import torch

import DeepSparseCoding.utils.loaders as loaders
from DeepSparseCoding.models.base_model import BaseModel
from DeepSparseCoding.modules.ensemble_module import EnsembleModule


class EnsembleModel(BaseModel, EnsembleModule):
    def setup(self, params, logger=None):
        """
        Setup required model components
        """
        super(EnsembleModel, self).setup(params, logger)
        self.setup_module(params)
        self.setup_optimizer()

    def setup_module(self, params):
        layer_names = [] # TODO: Make this submodule_name=model_type+layer_name is unique, not layer_name is unique
        for sub_index, subparams in enumerate(params.ensemble_params):
            layer_names.append(subparams.layer_name)
            assert len(set(layer_names)) == len(layer_names), (
                'The "layer_name" parameter must be unique for each module in the ensemble.')
            subparams.submodule_name = subparams.model_type + '_' + subparams.layer_name
            subparams.epoch_size = params.epoch_size
            subparams.batches_per_epoch = params.batches_per_epoch
            subparams.num_batches = params.num_batches
            #subparams.num_val_images = params.num_val_images
            #subparams.num_test_images = params.num_test_images
            if not hasattr(subparams, 'data_shape'): # TODO: This is a workaround for a dependency on data_shape in lca module
                subparams.data_shape = params.data_shape
        super(EnsembleModel, self).setup_ensemble_module(params)
        self.submodel_classes = []
        for ensemble_index, subparams in enumerate(self.params.ensemble_params):
            submodule_class = loaders.load_model_class(subparams.model_type)
            self.submodel_classes.append(submodule_class)
            if subparams.checkpoint_boot_log != '':
                checkpoint = self.get_checkpoint_from_log(subparams.checkpoint_boot_log)
                submodule = self.__getitem__(ensemble_index)
                module_state_dict_name = subparams.submodule_name+'_module_state_dict'
                if module_state_dict_name in checkpoint.keys(): # It was already in an ensemble
                    submodule.load_state_dict(checkpoint[module_state_dict_name])
                else: # it was trained on its own
                    submodule.load_state_dict(checkpoint['model_state_dict'])

    def setup_optimizer(self):
        for module in self:
            module.optimizer = self.get_optimizer(
                optimizer_params=module.params,
                trainable_variables=module.parameters())
            if module.params.checkpoint_boot_log != '':
                checkpoint = self.get_checkpoint_from_log(module.params.checkpoint_boot_log)
                module_state_dict_name = module.params.submodule_name+'_optimizer_state_dict'
                if module_state_dict_name in checkpoint.keys(): # It was already in an ensemble
                    module.optimizer.load_state_dict(checkpoint[module_state_dict_name])
                else: # it was trained on its own
                    module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'][0]) #TODO: For some reason this is a tuple of size 1 containing the dictionary. It should just be the dictionary
            module.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                module.optimizer,
                milestones=module.params.optimizer.milestones,
                gamma=module.params.optimizer.lr_decay_rate)

    def load_checkpoint(self, cp_file=None, load_optimizer=False):
        """
        Load checkpoint
        Keyword arguments:
          model_dir: [str] specifying the path to the checkpoint
        """
        if cp_file is None:
            cp_file = self.params.cp_latest_filename
        checkpoint = torch.load(cp_file)
        for module in self:
            module_state_dict_name = module.params.submodule_name+'_module_state_dict'
            if module_state_dict_name in checkpoint.keys(): # It was already in an ensemble
                module.load_state_dict(checkpoint[module_state_dict_name])
                _ = checkpoint.pop(module_state_dict_name, None)
            else: # it was trained on its own
                module.load_state_dict(checkpoint['model_state_dict'])
                _ = checkpoint.pop('optimizer_state_dict', None)
            if load_optimizer:
                module_state_dict_name = module.params.submodule_name+'_optimizer_state_dict'
                if module_state_dict_name in checkpoint.keys(): # It was already in an ensemble
                    module.optimizer.load_state_dict(checkpoint[module_state_dict_name])
                    _ = checkpoint.pop(module_state_dict_name, None)
                else:
                    module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    _ = checkpoint.pop('optimizer_state_dict', None)
        _ = checkpoint.pop('model_state_dict', None)
        training_status = pprint.pformat(checkpoint, compact=True)#, sort_dicts=True #TODO: Python 3.8 adds the sort_dicts parameter
        out_str = f'Loaded checkpoint from {cp_file} with the following stats:\n{training_status}'
        return out_str

    def preprocess_data(self, data):
        """
        We assume that only the first submodel will be preprocessing the input data
        """
        submodule = self.__getitem__(0)
        return self.submodel_classes[0].preprocess_data(submodule, data)

    def get_total_loss(self, input_tuple, ensemble_index):
        submodule = self.__getitem__(ensemble_index)
        submodel_class = self.submodel_classes[ensemble_index]
        return submodel_class.get_total_loss(submodule, input_tuple)

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(EnsembleModel, self).generate_update_dict(input_data,
            input_labels, batch_step)
        x = input_data.clone() # TODO: Do I need to clone it? If not then don't.
        for ensemble_index, submodel_class in enumerate(self.submodel_classes):
            submodule = self.__getitem__(ensemble_index)
            submodel_update_dict = submodel_class.generate_update_dict(submodule, x,
                input_labels, batch_step, update_dict=dict())
            for key, value in submodel_update_dict.items():
                if key not in ['epoch', 'batch_step']:
                    key = submodule.params.submodule_name + '_' + key
                update_dict[key] = value
            x = submodule.get_encodings(x)
        return update_dict
