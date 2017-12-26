import h5py
import numpy as np
from data.dataset import Dataset
import utils.data_processing as dp

class vanHateren(object):
  def __init__(self, img_dir, num_images=50, rand_state=np.random.RandomState()):
    full_img_data = self.extract_images(img_dir, num_images, rand_state=rand_state)[..., None]
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
      rand_state (optional) [obj] numpy random state object
      num_images (optional) [int] how many images to extract. Default (None) is all images.
      image_edge_size (optional) [int] how many pixels on an edge. Default (None) is full-size.
  """
  # Parse kwargs
  assert ("data_dir" in kwargs.keys()), ("function input must have 'data_dir' kwarg")
  data_dir = kwargs["data_dir"]
  rand_state = kwargs["rand_state"] if "rand_state" in kwargs.keys() else np.random.RandomState()
  num_images = int(kwargs["num_images"]) if "num_images"in kwargs.keys() else None
  image_edge_size = int(kwargs["image_edge_size"]) if "image_edge_size" in kwargs.keys() else None
  # Get data
  img_filename = data_dir+"/img/images_curated.h5" # pre-curated dataset
  vh_data = vanHateren(img_filename, num_images, rand_state)
  image_dataset = Dataset(vh_data.images, lbls=None, ignore_lbls=None, rand_state=rand_state)
  # Resize data
  if image_edge_size is not None:
    edge_scale = image_edge_size/image_dataset.shape[1] #vh has square images
    assert edge_scale <= 1.0, (
      "image_edge_size (%g) must be less than or equal to the original size (%g)."%(image_edge_size,
      image_dataset.shape[1]))
    scale_factor = [1.0, edge_scale, edge_scale, 1.0] # batch & channel don't get downsampled
    image_dataset.downsample(scale_factor, order=3)
  return {"train":image_dataset}
