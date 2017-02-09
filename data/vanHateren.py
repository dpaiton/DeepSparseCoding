import os
import h5py
import numpy as np
from data.dataset import Dataset

class vanHateren(object):
  def __init__(self,
    img_dir,
    patch_edge_size=None,
    rand_state=np.random.RandomState()):
    self.images = self.extract_images(img_dir, patch_edge_size)

  """
  Load in van Hateren dataset
  if patch_edge_size is specified, rebuild data array to be of sequential image
  patches
  """
  def extract_images(self, filename, patch_edge_size=None):
    with h5py.File(filename, "r") as f:
      full_img_data = np.array(f['van_hateren_good'], dtype=np.float32)
    if patch_edge_size is not None:
      (num_img, num_px_rows, num_px_cols) = full_img_data.shape
      if num_px_cols % patch_edge_size != 0: # crop columns
        crop_x = num_px_cols % patch_edge_size
        crop_edge = np.int(np.floor(crop_x/2.0))
        full_img_data = (
          full_img_data[:, crop_edge:num_px_cols-crop_edge, :])
        num_px_cols = full_img_data.shape[1]
      if num_px_rows % patch_edge_size != 0: # crop rows
        crop_y = num_px_rows % patch_edge_size
        crop_edge = np.int(np.floor(crop_y/2.0))
        full_img_data = (
          full_img_data[:, :, crop_edge:num_px_rows-crop_edge])
        num_px_rows = full_img_data.shape[2]
      num_px_img = num_px_rows * num_px_cols
      self.num_patches = int(num_px_img / patch_edge_size**2)
      # Tile column-wise, then row-wise
      data = np.asarray(np.split(full_img_data, num_px_cols/patch_edge_size, 2))
      data = np.asarray(np.split(data, num_px_rows/patch_edge_size, 2))
      data = np.transpose(np.reshape(np.transpose(data, axes=(3,4,0,1,2)),
        (patch_edge_size, patch_edge_size, -1)), axes=(2,0,1))
    else:
      data = full_img_data
      self.num_patches = 0
    return data

"""
Load van Hateren data and format as a Dataset object
inputs: kwargs [dict] containing keywords:
  data_dir [str] directory to van Hateren data
  whiten_images [bool] whether or not images should be whitened(not implemented)
  patch_edge_size [int] length of a patch edge if the images are to be broken up
    None (default) indicates that the full images should be used.
  rand_state [obj] numpy random state object
"""
def load_vanHateren(kwargs):
  assert ("data_dir" in kwargs.keys()), (
    "function input must have 'data_dir' key")
  data_dir = kwargs["data_dir"]
  whiten_images = (kwargs["whiten_images"]
    if "whiten_images" in kwargs.keys() else False)
  patch_edge_size = (kwargs["patch_edge_size"]
    if "patch_edge_size" in kwargs.keys() else None)
  rand_state = (kwargs["rand_state"]
    if "rand_state" in kwargs.keys() else np.random.RandomState())

  ## Training set
  img_filename = data_dir+os.sep+"images_curated.h5"
  vh_data = vanHateren(
    img_filename,
    patch_edge_size,
    rand_state=rand_state)
  images = Dataset(vh_data.images, None, None, rand_state=rand_state)
  setattr(images, "num_patches", vh_data.num_patches)
  return {"train":images}
