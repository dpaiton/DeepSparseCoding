import os
import sys
import unittest
from os.path import dirname as up

ROOT_DIR = up(up(up(os.path.realpath(__file__))))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

#import numpy as np
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa

import DeepSparseCoding.utils.loaders as loaders
#import DeepSparseCoding.utils.dataset_utils as datasets
#import DeepSparseCoding.utils.run_utils as run_utils


#class TestModels(unittest.TestCase):
#    def setUp(self):
#        self.dsc_dir = os.path.join(*[ROOT_DIR, 'DeepSparseCoding'])
#        self.model_list = loaders.get_model_list(self.dsc_dir)
#        self.test_params_file = os.path.join(*[self.dsc_dir, 'params', 'test_params.py'])
#
#    def test_LinfPGD_with_ensemble(self):
#        attack_params = {
#            'linfPGD':
#                {'rel_stepsize':1e-3,
#                'steps':3}} # max perturbation it can reach is 0.5
#        attack = fa.LinfPGD(**attack_params['linfPGD'])
#        epsilons = [0.3] # allowed perturbation size
#        params['ensemble'] = loaders.load_params_file(self.test_params_file, key='ensemble_params')
#        params['ensemble'].train_logs_per_epoch = None
#        params['ensemble'].shuffle_data = False
#        train_loader, val_loader, test_loader, data_params = datasets.load_dataset(params['ensemble'])
#        for key, value in data_params.items():
#            setattr(params['ensemble'], key, value)
#        models['ensemble'].setup(params['ensemble'])
#        models['ensemble'].to(params['ensemble'].device)
#        data, target = next(iter(train_loader))
#        train_data_batch = models['ensemble'].preprocess_data(data.to(params['ensemble'].device))
#        train_target_batch = target.to(params['ensemble'].device)
#        fmodel = PyTorchModel(model.eval(), bounds=(0, 1))
#        model_output = fmodel.forward()
#        adv_model_outputs, adv_images, success = attack(fmodel, train_data_batch, train_target_batch, epsilons=epsilons)
#
