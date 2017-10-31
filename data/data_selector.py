from data.vanHateren import load_vanHateren
from data.mnist import load_MNIST
from data.cifar import load_CIFAR
from data.synthetic import load_synthetic
from data.field import load_field

"""
Get function that returns the corresponding dataset
Inputs:
  kwargs: [dict] containing params returned from params/param_picker.py
    data_type: [str] containing the name of the dataset to return
      Current allowed values are obtained from get_dataset_list()
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
def get_data(kwargs):
  if kwargs["data_type"].lower() == "vanhateren":
    if "vanHateren" not in kwargs["data_dir"]:
      kwargs["data_dir"] += "/vanHateren/"
    dataset = load_vanHateren(kwargs)
  if kwargs["data_type"].lower() == "mnist":
    if "MNIST" not in kwargs["data_dir"]:
      kwargs["data_dir"] += "/MNIST/"
    dataset = load_MNIST(kwargs)
  if kwargs["data_type"].lower() == "cifar10" or kwargs["data_type"].lower() == "cifar100":
    if "CIFAR" not in kwargs["data_dir"]:
      kwargs["data_dir"] += "/CIFAR/"
    kwargs["num_classes"] = int(kwargs["data_type"][5:len(kwargs["data_type"])])
    dataset = load_CIFAR(kwargs)
  if kwargs["data_type"].lower() == "synthetic":
    assert "epoch_size" in kwargs.keys(), (
      "Params must include 'epoch_size'")
    dataset = load_synthetic(kwargs)
  if kwargs["data_type"].lower() == "field":
    dataset = load_field(kwargs)
  return dataset

def get_dataset_list():
  data_list = ["vanHateren", "field", "MNIST", "CIFAR10", "synthetic"]
  return data_list
