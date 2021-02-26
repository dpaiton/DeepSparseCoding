import os
import sys
import unittest
from os.path import dirname as up

ROOT_DIR = up(up(up(os.path.realpath(__file__))))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import numpy as np

import DeepSparseCoding.utils.loaders as loaders
import DeepSparseCoding.utils.dataset_utils as datasets
import DeepSparseCoding.utils.run_utils as run_utils


class TestModels(unittest.TestCase):
    def setUp(self):
        self.dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
        self.model_list = loaders.get_model_list(self.dsc_dir)
        self.test_params_file = os.path.join(*[self.dsc_dir, 'params', 'test_params.py'])

    ### TODO - add ability to test multiple options (e.g. 'conv' and 'fc') from test params
    def test_model_loading(self):
        for model_type in self.model_list:
            model_type = '_'.join(model_type.split('_')[:-1]) # remove '_model' at the end
            model = loaders.load_model(model_type)
            params = loaders.load_params_file(self.test_params_file, key=model_type+'_params')
            train_loader, val_loader, test_loader, data_params = datasets.load_dataset(params)
            for key, value in data_params.items():
                setattr(params, key, value)
            model.setup(params)


    ### TODO - more basic test to compute gradients per model
    #def test_gradients(self):
    #    for model_type in self.model_list:
    #        model_type = ''.join(model_type.split('_')[:-1]) # remove '_model' at the end
    #        model = loaders.load_model(model_type)

    ### TODO - test for gradient blocking
    #def test_get_module_encodings(self):
    #    """
    #    Test for gradient blocking in the get_module_encodings function

    #    construct test model1 & model2
    #    construct test ensemble model = model1 -> model2
    #    get encoding & grads for allow_grads={True, False}
    #    False: compare grads for model1 alone vs model1 in ensemble
    #    True: ensure that grad is different from model1 alone
    #        * Should also manually compute grads to compare?
    #    """
    #    # test should utilize  run_utils.get_module_encodings()


    def test_lca_ensemble_gradients(self):
        params = {}
        models = {}
        params['lca'] = loaders.load_params_file(self.test_params_file, key='lca_params')
        params['lca'].train_logs_per_epoch = None
        params['lca'].shuffle_data = False
        train_loader, val_loader, test_loader, data_params = datasets.load_dataset(params['lca'])
        for key, value in data_params.items():
            setattr(params['lca'], key, value)
        models['lca'] = loaders.load_model(params['lca'].model_type)
        models['lca'].setup(params['lca'])
        models['lca'].to(params['lca'].device)
        params['ensemble'] = loaders.load_params_file(self.test_params_file, key='ensemble_params')
        for key, value in data_params.items():
            setattr(params['ensemble'], key, value)
        err_msg = f'\ndata_shape={params["ensemble"].data_shape}'
        err_msg += f'\nnum_pixels={params["ensemble"].num_pixels}'
        err_msg += f'\nbatch_size={params["ensemble"].batch_size}'
        err_msg += f'\nepoch_size={params["ensemble"].epoch_size}'
        models['ensemble'] = loaders.load_model(params['ensemble'].model_type)
        models['ensemble'].setup(params['ensemble'])
        models['ensemble'].to(params['ensemble'].device)
        ensemble_state_dict = models['ensemble'].state_dict()
        ensemble_state_dict['lca.weight'] = models['lca'].weight.clone()
        models['ensemble'].load_state_dict(ensemble_state_dict)
        data, target = next(iter(train_loader))
        train_data_batch = models['lca'].preprocess_data(data.to(params['lca'].device))
        train_target_batch = target.to(params['lca'].device)
        models['lca'].optimizer.zero_grad()
        for submodel in models['ensemble']:
            submodel.optimizer.zero_grad()
        inputs = [train_data_batch] # only the first model acts on input
        for submodel in models['ensemble']:
            inputs.append(submodel.get_encodings(inputs[-1]).detach())
        lca_loss = models['lca'].get_total_loss((train_data_batch, train_target_batch))
        ensemble_losses = [models['ensemble'].get_total_loss((inputs[0], train_target_batch), 0)]
        ensemble_losses.append(models['ensemble'].get_total_loss((inputs[1], train_target_batch), 1))
        lca_loss.backward()
        ensemble_losses[0].backward()
        ensemble_losses[1].backward()
        lca_loss_val = lca_loss.cpu().detach().numpy()
        lca_w_grad = models['lca'].weight.grad.cpu().numpy()
        ensemble_loss_val = ensemble_losses[0].cpu().detach().numpy()
        ensemble_w_grad = models['ensemble'][0].weight.grad.cpu().numpy()
        assert lca_loss_val == ensemble_loss_val, (err_msg+'\n'
            +'Losses should be equal, but are lca={lca_loss_val} and ensemble={ensemble_loss_val}')
        assert np.all(lca_w_grad == ensemble_w_grad), (err_msg+'\nGrads should be equal, but are not.')
        lca_pre_train_w = models['lca'].weight.cpu().detach().numpy().copy()
        ensemble_pre_train_w = models['ensemble'][0].weight.cpu().detach().numpy().copy()
        run_utils.train_epoch(1, models['lca'], train_loader)
        run_utils.train_epoch(1, models['ensemble'], train_loader)
        lca_w = models['lca'].weight.cpu().detach().numpy().copy()
        ensemble_w = models['ensemble'][0].weight.cpu().detach().numpy().copy()
        assert np.all(lca_pre_train_w == ensemble_pre_train_w), (err_msg+'\n'
            +"lca & ensemble weights are not equal before one epoch of training")
        assert not np.all(lca_pre_train_w == lca_w), (err_msg+'\n'
            +"lca weights are not different from init after one epoch of training")
        assert not np.all(ensemble_pre_train_w == ensemble_w), (err_msg+'\n'
            +"ensemble weights are not different from init after one epoch of training")
        assert np.all(lca_w == ensemble_w), (err_msg+'\n'
            +"lca & ensemble weights are not equal after one epoch of training")
