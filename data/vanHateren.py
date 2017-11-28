import h5py
import numpy as np
from data.dataset import Dataset
import utils.data_processing as dp

class vanHateren(object):
  def __init__(self, img_dir, num_images=50, rand_state=np.random.RandomState()):
    full_img_data = self.extract_images(img_dir, num_images, rand_state=rand_state)
    self.num_images = num_images
    self.full_shape = full_img_data.shape
    self.images = full_img_data

  def extract_images(self, filename, num_images=None, rand_state=np.random.RandomState()):
    """
    Load in van Hateren dataset
    """
    with h5py.File(filename, "r") as f:
      full_img_data = np.array(f["van_hateren_good"], dtype=np.float32)
      if num_images is not None and num_images < full_img_data.shape[0] and num_images>0:
        im_keep_idx = rand_state.choice(full_img_data.shape[0], num_images, replace=False)
        full_img_data = full_img_data[im_keep_idx, ...]
    return full_img_data

def load_vanHateren(kwargs):
  """
  Load van Hateren data and format as a Dataset object
  Inputs:
    kwargs [dict] containing keywords:
      data_dir [str] directory to van Hateren data
      rand_state [obj] numpy random state object
  """
  assert ("data_dir" in kwargs.keys()), (
    "function input must have 'data_dir' key")
  data_dir = kwargs["data_dir"]
  rand_state = (np.random.RandomState(kwargs["rand_seed"])
    if "rand_seed" in kwargs.keys() else np.random.RandomState())
  # Number of images to pull from vh dataset
  num_images = (np.int(kwargs["num_images"])
    if "num_images"in kwargs.keys() else None)
  vectorize = kwargs["vectorize_data"] if "vectorize_data" in kwargs.keys() else True

  ## Training set
  img_filename = data_dir+"/img/images_curated.h5"
  vh_data = vanHateren(img_filename, num_images, rand_state=rand_state)
  images = Dataset(vh_data.images, lbls=None, ignore_lbls=None, vectorize=vectorize,
    rand_state=rand_state)

  ## TODO: This 'if' block is temporary - new code will rquire ndim == 3 after extraction,
  ##       so there is no need to check.
  image_edge_size = (int(np.floor(kwargs["image_edge_size"]))
    if "image_edge_size" in kwargs.keys() else None)
  if image_edge_size is not None:
    if images.ndim == 2:
      edge_scale = image_edge_size/images.shape[0] #vh has square images
      assert edge_scale < 1.0, (
        "image_edge_size (%g) must be greater than the original size (%g)."%(image_edge_size,
        images.shape[0]))
      scale_factor = [edge_scale, edge_scale]
    elif images.ndim > 2: #1st dim is expected to be batch
      edge_scale = image_edge_size/images.shape[1] #vh has square images
      assert edge_scale < 1.0, (
        "image_edge_size (%g) must be greater than the original size (%g)."%(image_edge_size,
        images.shape[0]))
      scale_factor = [1.0]+[edge_scale,]*(images.ndim-1)
    images.downsample(scale_factor, order=3)

  images.preprocess(kwargs)
  images.num_images = vh_data.num_images
  images.full_shape = vh_data.full_shape
  return {"train":images}
