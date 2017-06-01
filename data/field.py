import numpy as np
from data.dataset import Dataset
import utils.image_processing as ip

class field(object):
  def __init__(self, img_dir, num_examples=None, patch_edge_size=None,
    overlapping=None, var_thresh=None, rand_state=np.random.RandomState()):
    full_img_data = self.extract_images(img_dir)
    if all(param is not None for param in (num_examples, patch_edge_size,
      overlapping, var_thresh)):
      out_shape = (num_examples, patch_edge_size, patch_edge_size)
      self.images = ip.extract_patches(full_img_data, out_shape, overlapping,
        var_thresh, rand_state)
    else:
      self.images = full_img_data

  """
  Load in Field dataset
  """
  def extract_images(self, filename):
    full_img_data = np.load(filename)["IMAGES"].transpose((2,0,1))
    return full_img_data

"""
Load Field data and format as a Dataset object
Inputs:
  kwargs [dict] containing keywords:
    data_dir [str] directory to van Hateren data
    whiten_images [bool] whether or not images should be whitened(not implemented)
  rand_state [obj] numpy random state object
"""
def load_field(kwargs):
  assert ("data_dir" in kwargs.keys()), (
    "function input must have 'data_dir' key")
  data_dir = kwargs["data_dir"]
  whiten_images = (kwargs["whiten_images"]
    if "whiten_images" in kwargs.keys() else False)
  rand_state = (kwargs["rand_state"]
    if "rand_state" in kwargs.keys() else np.random.RandomState())
  patch_edge_size = (np.int(kwargs["patch_edge_size"])
    if "patch_edge_size" in kwargs.keys() else None)
  num_examples = (np.int(kwargs["epoch_size"])
    if "epoch_size" in kwargs.keys() else None)
  overlapping = (kwargs["overlapping_patches"]
    if "overlapping_patches" in kwargs.keys() else None)
  var_thresh = (kwargs["patch_variance_threshold"]
    if "patch_variance_threshold" in kwargs.keys() else None)
  vectorize = not kwargs["conv"] #conv models need a devectorized images

  ## Training set
  if whiten_images:
    img_filename = data_dir+"/field/IMAGES.npz"
  else:
    img_filename = data_dir+"/field/IMAGES_RAW.npz"
  field_data = field(img_filename, num_examples, patch_edge_size, overlapping,
    var_thresh, rand_state=rand_state)
  images = Dataset(field_data.images, lbls=None, ignore_lbls=None,
    vectorize=vectorize, rand_state=rand_state)
  return {"train":images}
