from data.vanHateren import load_vanHateren
from data.mnist import load_MNIST
from data.cifar import load_CIFAR
from data.synthetic import load_synthetic
from data.field import load_field

"""
Get function that returns the corresponding dataset
Inputs:
  dataset_type: [str] containing the name of the dataset to return
    Current allowed values are: vanHateren, mnist, laplacian
  params: [dict] containing params returned from params/param_picker.py
    data_dir (cifar, field, mnist, vanhateren)
    num_classes (cifar)
    num_val (cifar, mnist)
    num_labeled (cifar, mnist)
    rand_state (cifar, field, mnist, synthetic, vanhateren)
    conv (cifar, field, mnist, synthetic, vanhateren)
    whiten_images (field, vanhateren)
    patch_edge_size (field, synthetic, vanhateren)
    epoch_size (field, synthetic, vanhateren)
    overlapping_patches (field, vanhateren)
    patch_variance_threshold (field, vanhateren)
    dist_type (synthetic)
Outputs:
  dataset: [dataset] object containing the dataset
"""
def get_data(dataset_type, params):
  if dataset_type.lower() == "vanhateren":
    params["data_dir"] += "/vanHateren/"
    dataset = load_vanHateren(params)
  if dataset_type.lower() == "mnist":
    params["data_dir"] += "/MNIST/"
    dataset = load_MNIST(params)
  if dataset_type.lower() == "cifar10" or dataset_type.lower() == "cifar100":
    params["data_dir"] += "/CIFAR/"
    params["num_classes"] = int(dataset_type[5:len(dataset_type)])
    dataset = load_CIFAR(params)
  if dataset_type.lower() == "synthetic":
    assert "epoch_size" in params.keys(), (
      "Params must include 'epoch_size'")
    dataset = load_synthetic(params)
  if dataset_type.lower() == "field":
    dataset = load_field(params)
  return dataset

def get_dataset_list():
  data_list = ["vanHateren", "field", "MNIST", "CIFAR-10", "synthetic"]
  return data_list
