import numpy as np
import pickle
from data.dataset import Dataset
import utils.data_processing as dp

class CIFAR(object):
  def __init__(self, data_dir, num_val=0, num_labeled=50000, rand_state=np.random.RandomState()):
    self.num_labeled = num_labeled
    if num_val < 1:
      num_val = 0
    ## Extract images and labels
    train_val_images, d_train_val_labels = self.load_train_data(data_dir)
    train_val_images = train_val_images.astype(np.float32)/255.0
    self.num_classes = np.max(d_train_val_labels)+1
    self.test_images, d_test_labels = self.load_test_data(data_dir)
    self.test_images = self.test_images.astype(np.float32)/255.0
    self.test_labels = self.dense_to_one_hot(d_test_labels)
    ## Grab a random sample of images for the validation set
    tot_images = train_val_images.shape[0]
    if tot_images < num_val:
      num_val = tot_images
    if num_val > 0:
      val_indices = rand_state.choice(np.arange(tot_images, dtype=np.int32),
        size=num_val, replace=False)
      train_indices = np.setdiff1d(np.arange(tot_images, dtype=np.int32),
        val_indices).astype(np.int32)
    else:
      val_indices = None
      train_indices = np.arange(tot_images, dtype=np.int32)
    train_val_labels = self.dense_to_one_hot(d_train_val_labels)
    self.train_labels = train_val_labels[train_indices]
    self.val_labels = train_val_labels[val_indices]
    self.num_train_images = len(train_indices)
    self.num_val_images = num_val
    self.num_test_images = self.test_images.shape[0]
    self.train_images = train_val_images[train_indices]
    if val_indices is not None:
      self.val_images = train_val_images[val_indices]
    ## Construct list of images to be ignored
    self.ignore_labels = self.train_labels
    if self.num_labeled < self.num_train_images:
      ignore_idx_list = []
      for lbl in range(0, self.num_classes):
        lbl_loc = [idx
          for idx
          in np.arange(len(train_indices), dtype=np.int32)
          if train_val_labels[train_indices[idx]] == lbl]
        ignore_idx_list.extend(rand_state.choice(lbl_loc,
          size=int(len(lbl_loc) - (self.num_labeled/float(self.num_classes))),
          replace=False).tolist())
      ignore_indices = np.array(ignore_idx_list, dtype=np.int32)
      self.ignore_labels[ignore_indices] = 0

  def load_train_data(self, data_dir):
    """Load train image batches from file"""
    data_list = []
    train_label_list = []
    for batch_id in range(1,6):
      data_loc = data_dir+"/data_batch_{}".format(batch_id)
      (data, labels) = self.unpickle(data_loc)
      train_label_list += [labels]
      data_list.append(data.reshape(data.shape[0], 3, 32, 32).transpose((0,2,3,1)))
    train_data = np.vstack(data_list)
    train_labels = np.hstack(train_label_list)
    return train_data, train_labels

  def load_test_data(self, data_dir):
    """Load test image batches from file"""
    (data, labels) = self.unpickle(data_dir+"/test_batch")
    test_labels = labels
    test_data = data.reshape(data.shape[0], 3, 32, 32).transpose((0,2,3,1))
    return test_data, test_labels

  def unpickle(self, filename):
    """Load byte data from file"""
    with open(filename, 'rb') as f:
      cifar = pickle.load(f, encoding="bytes")
    return (cifar[b"data"], np.array(cifar[b"labels"]))

  def dense_to_one_hot(self, labels_dense):
    """Convert vector of dense labels to a matrix of one-hot labels"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels, dtype=np.int32) * self.num_classes
    labels_one_hot = np.zeros((num_labels, self.num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def load_CIFAR(kwargs):
  assert ("data_dir" in kwargs.keys()), ("load_CIFAR function input must have 'data_dir' key")
  #assert ("num_classes" in kwargs.keys()), ("load_CIFAR function input must have 'num_classes' key")
  data_dir = kwargs["data_dir"]
  num_val = kwargs["num_val"] if "num_val" in kwargs.keys() else 10000
  num_labeled = kwargs["num_labeled"] if "num_labeled" in kwargs.keys() else 50000
  rand_state = kwargs["rand_state"] if "rand_state" in kwargs.keys() else np.random.RandomState()
  data_dir = data_dir+"/cifar-10-batches-py/" # TODO: implement CIFAR100
  #if kwargs["num_classes"] == 10:
  #  data_dir = data_dir+"/cifar-10-batches-py/"
  #elif kwargs["num_classes"] == 100:
  #  assert False, "CIFAR-100 is not supported"
  #else:
  #  assert False, (
  #  "'num_classes' key must be 10 or 100 for CIFAR-10 or CIFAR-100")
  train_val_test = CIFAR(data_dir, num_val, num_labeled, rand_state)
  train = Dataset(train_val_test.train_images, train_val_test.train_labels,
    train_val_test.ignore_labels, rand_state)
  val = Dataset(train_val_test.val_images, train_val_test.val_labels, None, rand_state)
  test = Dataset(train_val_test.test_images, train_val_test.test_labels, None, rand_state)
  return {"train":train, "val":val, "test":test}
