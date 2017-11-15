import numpy as np
from data.dataset import Dataset
import utils.data_processing as dp

class field(object):
  def __init__(self, img_dir, num_images=None, rand_state=np.random.RandomState()):
    full_img_data = self.extract_images(img_dir)
    self.num_images = num_images
    self.full_shape = full_img_data.shape
    self.images = full_img_data

  def extract_images(self, filename):
    """
    Load in Field dataset
    """
    full_img_data = np.load(filename)["IMAGES"].transpose((2,0,1))
    return full_img_data

def load_field(kwargs):
  """
  Load Field data and format as a Dataset object
  Inputs:
    kwargs [dict] containing keywords:
      data_dir [str] directory to van Hateren data
      whiten_data [bool] whether or not images should be whitened(not implemented)
    rand_state [obj] numpy random state object
  """
  assert ("data_dir" in kwargs.keys()), (
    "function input must have 'data_dir' key")
  data_dir = kwargs["data_dir"]
  rand_state = (kwargs["rand_state"]
    if "rand_state" in kwargs.keys() else np.random.RandomState())
  num_images = (np.int(kwargs["num_images"])
    if "num_images" in kwargs.keys() else None)
  vectorize = kwargs["vectorize_data"] if "vectorize_data" in kwargs.keys() else True

  img_filename = data_dir+"/field/IMAGES_RAW.npz" # NOTE: IMAGES.npz has whitened data
  field_data = field(img_filename, num_images, rand_state=rand_state)
  images = Dataset(field_data.images, lbls=None, ignore_lbls=None,
    vectorize=vectorize, rand_state=rand_state)
  images.preprocess(kwargs)
  images.num_images = field_data.num_iamges
  images.full_shape = field_data.full_shape
  return {"train":images}
