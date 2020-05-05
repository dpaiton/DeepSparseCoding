import torch
from torchvision import datasets, transforms
import os
import numpy as np

import DeepSparseCoding.utils.data_processing as dp

class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def load_dataset(params):
    if(params.dataset.lower() == 'mnist'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)) # channels last
            ]
        if params.standardize_data:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.standardize(x, eps=params.eps)[0]))
        # Load dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=params.data_dir, train=True, download=True,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=params.data_dir, train=False, download=True,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
    elif(params.dataset.lower() == 'dsprites'):
        root = 'data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
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
    else:
        assert False, (f'Supported datasets are ["mnist"], not {dataset_name}')
    params.epoch_size = len(train_loader.dataset)
    if(not hasattr(params, 'num_val_images')):
        if test_loader is not None:
            params.num_val_images = len(test_loader.dataset)
    if(not hasattr(params, 'num_test_images')):
        if test_loader is not None:
            params.num_test_images = len(test_loader.dataset)
    params.data_shape = list(next(iter(train_loader))[0].shape)[1:]
    return (train_loader, val_loader, test_loader, params)
