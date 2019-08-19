import numpy as np
import pickle
from data.dataset import Dataset
from data.cifar import CIFAR
import utils.data_processing as dp
import pdb
import skimage

class CIFAR_GRAY(CIFAR):
  def load_train_data(self, data_dir):
    train_data, train_labels = super().load_train_data(data_dir)
    train_data = dp.rgb2gray(train_data).astype(np.uint8)
    #train_data = dp.resize_images(train_data, [28, 28]).astype(np.uint8)
    return train_data, train_labels

  def load_test_data(self, data_dir):
    test_data, test_labels = super().load_test_data(data_dir)
    test_data = dp.rgb2gray(test_data).astype(np.uint8)
    #test_data = dp.resize_images(test_data, [28, 28]).astype(np.uint8)
    return test_data, test_labels

def load_CIFAR_GRAY(params):
  assert ("data_dir" in params.__dict__.keys()), ("load_CIFAR_GRAY function input must have 'data_dir' key")
  #assert ("num_classes" in params.__dict__.keys()), ("load_CIFAR function input must have 'num_classes' key")
  data_dir = params.data_dir
  num_val = params.num_val if hasattr(params, "num_val") else 10000
  num_labeled = params.num_labeled if hasattr(params, "num_labeled") else 50000
  if hasattr(params, "rand_state"):
    rand_state = params.rand_state
  else:
    assert hasattr(params, "rand_seed"), ("Params must specify a random state or seed")
    rand_state = np.random.RandomState(params.rand_seed)
  data_dir = data_dir+"/cifar-10-batches-py/" # TODO: implement CIFAR100
  #if params.num_classes == 10:
  #  data_dir = data_dir+"/cifar-10-batches-py/"
  #elif params.num_classes == 100:
  #  assert False, "CIFAR-100 is not supported"
  #else:
  #  assert False, (
  #  "'num_classes' key must be 10 or 100 for CIFAR-10 or CIFAR-100")
  train_val_test = CIFAR_GRAY(data_dir, num_val, num_labeled, rand_state)

  train = Dataset(train_val_test.train_images, train_val_test.train_labels,
    train_val_test.ignore_labels, rand_state)
  val = Dataset(train_val_test.val_images, train_val_test.val_labels, None, rand_state)
  test = Dataset(train_val_test.test_images, train_val_test.test_labels, None, rand_state)


  return {"train":train, "val":val, "test":test}


