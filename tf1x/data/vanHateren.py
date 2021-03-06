import h5py
import numpy as np

from DeepSparseCoding.tf1x.data.dataset import Dataset
import DeepSparseCoding.tf1x.utils.data_processing as dp

class vanHateren(object):
  def __init__(self, image_dir, num_images=50, rand_state=np.random.RandomState()):
    full_img_data = self.extract_images(image_dir, num_images, rand_state)[..., None]
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

def load_vanHateren(params):
  """
  Load van Hateren data and format as a Dataset object
  Inputs:
    params [obj] containing attributes:
      data_dir [str] directory to van Hateren data
      rand_state (optional) [obj] numpy random state object
      num_images (optional) [int] how many images to extract. Default (None) is all images.
      image_edge_size (optional) [int] how many pixels on an edge. Default (None) is full-size.
  """
  # Parse params
  assert hasattr(params, "data_dir"), ("function input must have 'data_dir' kwarg")
  data_dir = params.data_dir
  if hasattr(params, "rand_state"):
    rand_state = params.rand_state
  else:
    #assert hasattr(params, "rand_seed"), ("Params must specify a random state or seed")
    if hasattr(params, "rand_seed"):
      rand_state = np.random.RandomState(params.rand_seed)
    else:
      rand_state = np.random.RandomState(None)
      print("WARNING: Params did not specify a random state or seed")
  num_images = int(params.num_images) if hasattr(params, "num_images") else None
  image_edge_size = int(params.image_edge_size) if hasattr(params, "image_edge_size") else None
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
