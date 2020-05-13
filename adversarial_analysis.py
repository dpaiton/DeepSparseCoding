import os
import sys

parent_path = os.path.dirname(os.getcwd())
if parent_path not in sys.path: sys.path.append(parent_path)

import numpy as np
import proplot as plot
import torch

from DeepSparseCoding.utils.file_utils import Logger
import DeepSparseCoding.utils.loaders as loaders
import DeepSparseCoding.utils.run_utils as run_utils
import DeepSparseCoding.utils.dataset_utils as dataset_utils
import DeepSparseCoding.utils.run_utils as ru
import DeepSparseCoding.utils.plot_functions as pf

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa


workspace_dir = os.path.expanduser('~')+'/Work/'
log_files = [
    os.path.join(*[workspace_dir, 'Torch_projects', 'mlp_768_mnist', 'logfiles', 'mlp_768_mnist_v0.log']),
    os.path.join(*[workspace_dir, 'Torch_projects', 'lca_768_mlp_mnist', 'logfiles', 'lca_768_mlp_mnist_v0.log'])
    ]

cp_latest_filenames = [
    os.path.join(*[workspace_dir,'Torch_projects', 'mlp_768_mnist', 'checkpoints', 'mlp_768_mnist_latest_checkpoint_v0.pt']),
    os.path.join(*[workspace_dir, 'Torch_projects', 'lca_768_mlp_mnist', 'checkpoints', 'lca_768_mlp_mnist_latest_checkpoint_v0.pt'])
    ]

attack_params = {
    'linfPGD':
        {'rel_stepsize':1e-3,
        'steps':int(1/(2*1e-3))}} # max perturbation it can reach is 0.5

attacks = [
    fa.FGSM(),
    fa.LinfPGD(**attack_params['linfPGD']),
    #fa.LinfBasicIterativeAttack(),
    #fa.LinfAdditiveUniformNoiseAttack(),
    #fa.LinfDeepFoolAttack(),
]

epsilons = [ # allowed perturbation size
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    #0.8,
    #1.0
]

num_models = len(log_files)
for model_index in range(num_models):
    logger = Logger(log_files[model_index], overwrite=False)
    log_text = logger.load_file()
    params = logger.read_params(log_text)[-1]
    params.cp_latest_filename = cp_latest_filenames[model_index]
    train_loader, val_loader, test_loader, data_params = dataset_utils.load_dataset(params)
    for key, value in data_params.items():
        setattr(params, key, value)
    model = loaders.load_model(params.model_type, params.lib_root_dir)
    model.setup(params, logger)
    model.params.analysis_out_dir = os.path.join(
        *[model.params.model_out_dir, 'analysis', model.params.version])
    model.params.analysis_save_dir = os.path.join(model.params.analysis_out_dir, 'savefiles')
    if not os.path.exists(model.params.analysis_save_dir):
        os.makedirs(model.params.analysis_save_dir)
    model.to(params.device)
    model.load_checkpoint()
    fmodel = PyTorchModel(model.eval(), bounds=(0, 1))
    print('\n', '~' * 79)
    num_batches =  len(test_loader.dataset) // model.params.batch_size
    attack_success = np.zeros(
            (len(attacks), len(epsilons), num_batches, model.params.batch_size), dtype=np.bool)
    for batch_index, (data, target) in enumerate(test_loader):
        #if batch_index < 10:
        data = model.preprocess_data(data.to(model.params.device))
        target = target.to(model.params.device)
        images, labels = ep.astensors(*(data, target))
        del data; del target
        print(f'Model type: {model.params.model_type} [{model_index+1} out of {len(log_files)}]')
        print(f'Batch {batch_index+1} out of {num_batches}')
        print(f'accuracy {accuracy(fmodel, images, labels)}')
        for attack_index, attack in enumerate(attacks):
            _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
            assert success.shape == (len(epsilons), len(images))
            success_ = success.numpy()
            assert success_.dtype == np.bool
            attack_success[attack_index, :, batch_index, :] = success_
            print('\n', attack)
            print('  ', 1.0 - success_.mean(axis=-1).round(2))
        robust_accuracy = 1.0 - attack_success[:, :, batch_index, :].max(axis=0).mean(axis=-1)
        print('\n', '-' * 79, '\n')
        print('worst case (best attack per-sample)')
        print('  ', robust_accuracy.round(2))
        print('-' * 79)
    attack_success = attack_success.reshape(
        (len(attacks), len(epsilons), num_batches*model.params.batch_size))
    attack_types = [str(type(attack)).split('.')[-1][:-2] for attack in attacks]
    output_filename = os.path.join(model.params.analysis_save_dir,
        f'linf_adversarial_analysis.npz')
    out_dict = {
        'adversarial_analysis':attack_success,
        'attack_types':attack_types,
        'epsilons':epsilons,
        'attack_params':attack_params}
    np.savez(output_filename, data=out_dict)
