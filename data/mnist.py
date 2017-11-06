import numpy as np
import gzip
from data.dataset import Dataset

class MNIST(object):
  def __init__(self,
    img_dir,
    lbl_dir,
    num_val=0,
    num_labeled=50000,
    rand_state=np.random.RandomState()):

    self.num_labeled = num_labeled
    if num_val < 1:
      num_val = 0
    self.num_classes = 10 # 10 MNIST classes

    ## Extract images
    self.images = self.extract_images(img_dir)
    self.labels = self.extract_labels(lbl_dir)
    self.labels = self.dense_to_one_hot(self.labels)
    assert self.images.shape[0] == self.labels.shape[0], (
      "Error: %g images and %g labels"%(self.images.shape[0],
      self.labels.shape[0]))

    ## Grab a random sample of images for the validation set
    tot_images = self.images.shape[0]
    if tot_images < num_val:
      num_val = tot_images
    if num_val > 0:
      self.val_indices = rand_state.choice(np.arange(tot_images,
        dtype=np.int32), size=num_val, replace=False)
      self.train_indices = np.setdiff1d(np.arange(tot_images, dtype=np.int32),
        self.val_indices).astype(np.int32)
    else:
      self.val_indices = None
      self.train_indices = np.arange(tot_images, dtype=np.int32)

    self.num_train_images = len(self.train_indices)

    ## Construct list of images to be ignored
    if self.num_labeled < self.num_train_images:
      ignore_idx_list = []
      for lbl in range(0, self.num_classes):
        lbl_loc = [idx
          for idx
          in np.arange(len(self.train_indices), dtype=np.int32)
          if self.labels[self.train_indices[idx]] == lbl]
        ignore_idx_list.extend(rand_state.choice(lbl_loc,
          size=int(len(lbl_loc) - (self.num_labeled/float(self.num_classes))),
          replace=False).tolist())
      self.ignore_indices = np.array(ignore_idx_list, dtype=np.int32)
    else:
      self.ignore_indices = None

  def dense_to_one_hot(self, labels_dense):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels, dtype=np.int32) * self.num_classes
    labels_one_hot = np.zeros((num_labels, self.num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

  def read_4B(self, bytestream):
    dt = np.dtype(np.uint32).newbyteorder("B") #big-endian byte order-MSB first
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

  def read_img_header(self, bytestream):
    _ = self.read_4B(bytestream)
    num_img = self.read_4B(bytestream)
    img_rows = self.read_4B(bytestream)
    img_cols = self.read_4B(bytestream)
    return (num_img, img_rows, img_cols)

  def read_lbl_header(self, bytestream):
    _ = self.read_4B(bytestream)
    return self.read_4B(bytestream)

  """Extract MNIST zip file, reshape, normalize"""
  def extract_images(self, filename):
    with open(filename, "rb") as f:
      with gzip.GzipFile(fileobj=f) as bytestream:
        num_img, img_rows, img_cols = self.read_img_header(bytestream)
        buf = bytestream.read(num_img*img_rows*img_cols)
        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(num_img, img_rows, img_cols, 1)
        images = images.astype(np.float32)
        images /= 255.0
        return images

  def extract_labels(self, filename):
    with open(filename, "rb") as f:
      with gzip.GzipFile(fileobj=f) as bytestream:
        num_labels = self.read_lbl_header(bytestream)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels.astype(np.int32)

"""
Load MNIST data and format as a Dataset object
inputs: kwargs [dict] containing keywords:
  data_dir [str] directory to MNIST data
  num_val [int] (10000) number of validation images
  num_labeled [int] (50000) number of labeled images
  rand_state [obj] (np.random.RandomState()) numpy random state object
"""
def load_MNIST(kwargs):
  assert ("data_dir" in kwargs.keys()), (
    "function input must have 'data_dir' key")
  data_dir = kwargs["data_dir"]
  num_val = kwargs["num_val"] if "num_val" in kwargs.keys() else 10000
  num_labeled = (kwargs["num_labeled"]
    if "num_labeled" in kwargs.keys() else 50000)
  rand_state = (kwargs["rand_state"]
    if "rand_state" in kwargs.keys() else np.random.RandomState())
  vectorize = kwargs["vectorize"] if "vectorize" in kwargs.keys() else True

  ## Training set
  train_img_filename = data_dir+"/train-images-idx3-ubyte.gz"
  train_lbl_filename = data_dir+"/train-labels-idx1-ubyte.gz"
  train_val = MNIST(
    train_img_filename,
    train_lbl_filename,
    num_val=num_val,
    num_labeled=num_labeled,
    rand_state=rand_state)
  train_images = train_val.images[train_val.train_indices, ...]
  train_lbls = train_val.labels[train_val.train_indices, ...]
  train_ignore_lbls = train_lbls.copy()
  if train_val.ignore_indices is not None:
    train_ignore_lbls[train_val.ignore_indices, ...] = 0
  train = Dataset(train_images, train_lbls, train_ignore_lbls, vectorize,
    rand_state=rand_state)

  ## Validation set
  if num_val > 0:
    val_images = train_val.images[train_val.val_indices]
    val_lbls = train_val.labels[train_val.val_indices]
    val_ignore_lbls = val_lbls.copy()
    val = Dataset(val_images, val_lbls, val_ignore_lbls, vectorize,
      rand_state=rand_state)
  else:
    val = None

  ## Test set
  test_img_filename = data_dir+"/t10k-images-idx3-ubyte.gz"
  test_lbl_filename = data_dir+"/t10k-labels-idx1-ubyte.gz"
  test = MNIST(
    test_img_filename,
    test_lbl_filename,
    num_val=0,
    num_labeled=10000,
    rand_state=rand_state)
  test_images = test.images
  test_lbls = test.labels
  test_ignore_lbls = test_lbls.copy()
  test = Dataset(test_images, test_lbls, test_ignore_lbls, vectorize,
    rand_state=rand_state)

  return {"train":train, "val":val, "test":test}
