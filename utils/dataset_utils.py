import os
import sys
from os.path import dirname as up

ROOT_DIR = up(up(up(os.path.realpath(__file__))))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import numpy as np
import torch
from torchvision import transforms
import torchvision.datasets

import DeepSparseCoding.utils.data_processing as dp
import DeepSparseCoding.datasets.synthetic as synthetic


class FastMNIST(torchvision.datasets.MNIST):
    """
    The torchvision MNIST dataset has additional overhead that slows it down.
    This loads the entire dataset onto the specified device at init, resulting in a considerable speedup
    """
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', 'cpu')
        super().__init__(*args, **kwargs)
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(-1).float().div(255)
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def load_dataset(params):
    new_params = {}
    if(params.dataset.lower() == 'mnist'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
        ]
        if params.standardize_data:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.standardize(x, eps=params.eps)[0]))
        if params.rescale_data_to_one:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.rescale_data_to_one(x, eps=params.eps, samplewise=True)[0]))
        kwargs = {
            'root':params.data_dir,
            'download':False,
            'transform':transforms.Compose(preprocessing_pipeline)
        }
        if hasattr(params, 'fast_mnist') and params.fast_mnist:
            kwargs['device'] = params.device
            kwargs['train'] = True
            train_loader = torch.utils.data.DataLoader(
                FastMNIST(**kwargs), batch_size=params.batch_size,
                shuffle=params.shuffle_data, num_workers=0, pin_memory=False)
            kwargs['train'] = False
            val_loader = None
            test_loader = torch.utils.data.DataLoader(
                FastMNIST(**kwargs), batch_size=params.batch_size,
                shuffle=params.shuffle_data, num_workers=0, pin_memory=False)
        else:
            kwargs['train'] = True
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(**kwargs), batch_size=params.batch_size,
                shuffle=params.shuffle_data, num_workers=0, pin_memory=True)
            kwargs['train'] = False
            val_loader = None
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(**kwargs), batch_size=params.batch_size,
                shuffle=params.shuffle_data, num_workers=0, pin_memory=True)

    elif(params.dataset.lower() == 'cifar10'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
        ]
        kwargs = {
            'root': os.path.join(*[params.data_dir,'cifar10']),
            'download': False,
            'train': True,
            'transform': transforms.Compose(preprocessing_pipeline)
        }
        if params.center_dataset:
            dataset = torchvision.datasets.CIFAR10(**kwargs)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,
                shuffle=False, num_workers=0, pin_memory=True)
            dataset_mean_image = dp.get_mean_from_dataloader(data_loader)
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: x - dataset_mean_image))
        if params.standardize_data:
            preprocessing_pipeline.append(
                transforms.Lambda(
                    lambda x: dp.standardize(x, eps=params.eps, samplewise=True, batch_size=params.batch_size)[0]
                )
            )
        if params.rescale_data_to_one:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.rescale_data_to_one(x, eps=params.eps, samplewise=True)[0]))
        kwargs['transform'] = transforms.Compose(preprocessing_pipeline)
        kwargs['train'] = True
        dataset = torchvision.datasets.CIFAR10(**kwargs)
        kwargs['train'] = False
        testset = torchvision.datasets.CIFAR10(**kwargs)
        num_train = len(dataset) - params.num_validation
        trainset, valset = torch.utils.data.random_split(dataset,
            [num_train, params.num_validation])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
            shuffle=params.shuffle_data, num_workers=0, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.batch_size,
            shuffle=False, num_workers=0, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
            shuffle=False, num_workers=0, pin_memory=True)

    elif(params.dataset.lower() == 'dsprites'):
        root = os.path.join(*[params.data_dir])
        dsprites_file = os.path.join(*[root, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'])
        if not os.path.exists(dsprites_file):
            import subprocess
            print(f'Now downloading the dsprites-dataset to {root}/dsprites')
            subprocess.call(['./scripts/download_dsprites.sh', f'{root}'])
            print('Finished')
        data = np.load(dsprites_file, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
        train_data = dset(**train_kwargs)
        train_loader = torch.utils.data.DataLoader(train_data,
            batch_size=params.batch_size,
            shuffle=params.shuffle_data,
            num_workers=0,
            pin_memory=False)
        val_loader = None
        test_loader = None

    elif(params.dataset.lower() == 'synthetic'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
        ]
        train_loader = torch.utils.data.DataLoader(
            synthetic.SyntheticImages(params.epoch_size, params.data_edge_size, params.dist_type,
            params.rand_state, params.num_classes,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = None
        new_params["num_pixels"] = params.data_edge_size**2

    else:
        assert False, (f'Supported datasets are ["mnist", "dsprites", "synthetic"], not {dataset_name}')
    new_params = {}
    new_params['epoch_size'] = len(train_loader.dataset)
    if(not hasattr(params, 'num_val_images')):
        if val_loader is None:
            new_params['num_val_images'] = 0
        else:
            new_params['num_val_images'] = len(val_loader.dataset)
    if(not hasattr(params, 'num_test_images')):
        if test_loader is None:
            new_params['num_test_images'] = 0
        else:
            new_params['num_test_images'] = len(test_loader.dataset)
    new_params['data_shape'] = list(next(iter(train_loader))[0].shape)[1:]
    new_params['num_pixels'] = np.prod(new_params['data_shape'])
    return (train_loader, val_loader, test_loader, new_params)
