import h5py
import numpy as np
from data.dataset import Dataset
import utils.image_processing as ip

class vanHateren(object):
  def __init__(self, img_dir, whiten_data=False, contrast_normalize=False,
    num_images=50, num_examples=None, patch_edge_size=None, overlapping=None,
    var_thresh=None, rand_state=np.random.RandomState()):
    full_img_data = self.extract_images(img_dir, num_images,
      rand_state=rand_state)
    full_img_data = ip.downsample_data(full_img_data, factor=[1, 0.5, 0.5],
      order=2)

    if whiten_data:
      full_img_data = ip.whiten_data(full_img_data, method="FT")
    else:
      full_img_data = ip.standardize_data(full_img_data)
    if contrast_normalize:
      full_img_data = ip.contrast_normalize(full_img_data)
    if all(param is not None for param in (num_examples, patch_edge_size,
      overlapping, var_thresh)):
      out_shape = (num_examples, patch_edge_size, patch_edge_size)
      self.images = ip.extract_patches(full_img_data, out_shape, overlapping,
        var_thresh, rand_state)
    else:
      self.images = full_img_data

  """
  Load in van Hateren dataset
  """
  def extract_images(self, filename, num_images=50,
    rand_state=np.random.RandomState()):
    with h5py.File(filename, "r") as f:
      full_img_data = np.array(f["van_hateren_good"], dtype=np.float32)
      im_keep_idx = rand_state.choice(full_img_data.shape[0], num_images,
        replace=False)
      full_img_data = full_img_data[im_keep_idx, ...]
    return full_img_data

"""
Load van Hateren data and format as a Dataset object
Inputs:
  kwargs [dict] containing keywords:
    data_dir [str] directory to van Hateren data
    whiten_images [bool] whether or not images should be whitened(not implemented)
  rand_state [obj] numpy random state object
"""
def load_vanHateren(kwargs):
  assert ("data_dir" in kwargs.keys()), (
    "function input must have 'data_dir' key")
  data_dir = kwargs["data_dir"]
  whiten_images = (kwargs["whiten_images"]
    if "whiten_images" in kwargs.keys() else False)
  contrast_normalize = (kwargs["contrast_normalize"]
    if "contrast_normalize" in kwargs.keys() else False)
  rand_state = (kwargs["rand_state"]
    if "rand_state" in kwargs.keys() else np.random.RandomState())
  patch_edge_size = (np.int(kwargs["patch_edge_size"])
    if "patch_edge_size" in kwargs.keys() else None)
  num_images = (np.int(kwargs["num_images"])
    if "num_images"in kwargs.keys() else 50)
  num_examples = (np.int(kwargs["epoch_size"])
    if "epoch_size" in kwargs.keys() else None)
  overlapping = (kwargs["overlapping_patches"]
    if "overlapping_patches" in kwargs.keys() else None)
  var_thresh = (kwargs["patch_variance_threshold"]
    if "patch_variance_threshold" in kwargs.keys() else None)
  vectorize = not kwargs["conv"] #conv models need a devectorized images

  ## Training set
  img_filename = data_dir+"/img/images_curated.h5"
  vh_data = vanHateren(img_filename, whiten_images, contrast_normalize, num_images,
    num_examples, patch_edge_size, overlapping, var_thresh, rand_state=rand_state)
  images = Dataset(vh_data.images, lbls=None, ignore_lbls=None,
    vectorize=vectorize, rand_state=rand_state)
  return {"train":images}
