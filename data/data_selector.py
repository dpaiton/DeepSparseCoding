from data.vanHateren import load_vanHateren
from data.mnist import load_MNIST
from data.cifar import load_CIFAR
from data.synthetic import load_synthetic
from data.field import load_field
from data.tinyImages import load_tinyImages

"""
Get function that returns the corresponding dataset
Inputs:
  params: [obj] containing params returned from params/param_picker.py
    data_type: [str] containing the name of the dataset to return
      Current allowed values are obtained from get_dataset_list()
    data_dir (cifar, field, mnist, vanhateren)
    num_classes (cifar)
    num_val (cifar, mnist)
    num_labeled (cifar, mnist)
    rand_state (cifar, field, mnist, synthetic, vanhateren)
    vectorize_data (cifar, field, mnist, synthetic, vanhateren)
    whiten_data (field, vanhateren)
    patch_edge_size (field, synthetic, vanhateren)
    epoch_size (field, synthetic, vanhateren)
    overlapping_patches (field, vanhateren)
    patch_variance_threshold (field, vanhateren)
    dist_type (synthetic)
Outputs:
  dataset: [dataset] object containing the dataset
"""
def get_data(params):
  assert "data_type" in params.__dict__.keys(), ("params must include data_type")
  if params.data_type.lower() == "vanhateren":
    if "vanHateren" not in params.data_dir:
      params.data_dir += "/vanHateren/"
    dataset = load_vanHateren(params)
  elif params.data_type.lower() == "mnist":
    if "MNIST" not in params.data_dir:
      params.data_dir += "/MNIST/"
    dataset = load_MNIST(params)
  elif params.data_type.lower() == "cifar10" or params.data_type.lower() == "cifar100":
    if "CIFAR" not in params.data_dir:
      params.data_dir += "/CIFAR/"
    params.num_classes = int(params.data_type[5:len(params.data_type)])
    dataset = load_CIFAR(params)
  elif params.data_type.lower() == "synthetic":
    assert "epoch_size" in params.__dict__.keys(), ("Params must include 'epoch_size'")
    dataset = load_synthetic(params)
  elif params.data_type.lower() == "field":
    dataset = load_field(params)
  elif params.data_type.lower() == "tinyimages":
    if "TinyImages" not in params.data_dir:
      params.data_dir += "/TinyImages/"
    dataset = load_tinyImages(params)
  else:
    data_list = get_dataset_list()
    assert False, ("data_type "+str(params.data_type)+" is not supported. "
      +"Supported data_types are: \n"+" ".join(data_list))
  return dataset

def get_data_from_string(dataset_string, directory_string):
  class params(object):
    def __init__(self):
      self.data_type = dataset_string
      self.data_dir = directory_string
  return get_data(params())

def get_dataset_list():
  data_list = ["vanHateren", "field", "MNIST", "CIFAR10", "tinyImages", "synthetic"]
  return data_list
