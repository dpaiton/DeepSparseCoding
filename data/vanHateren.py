import os
import h5py
from data.dataset import Dataset

class vanHateren(object):
  def __init__(self,
    img_dir,
    patch_edge_size=None,
    rand_state=np.random.RandomState()):
    self.images = self.extract_images(img_dir, patch_edge_size)

  """
  load in van hateren dataset
  if patch_edge_size is specified, rebuild data array to be of sequential image
  patches
  """
  def extract_images(self, filename, patch_edge_size=None):
    with h5py.File(filename, "r") as f:
      full_img_data = np.array(f['van_hateren_good'], dtype=np.float32)
    if patch_edge_size is not None:
      (num_img, num_px_rows, num_px_cols) = full_img_data.shape
      num_img_px = num_px_rows * num_px_cols
      assert np.sqrt(num_img_px) % patch_edge_size == 0, (
        "The number of image edge pixels % the patch edge size must be 0.")
      self.num_patches = int(num_img_px / patch_edge_size**2)
      full_img_data = np.reshape(full_img_data, (num_img, num_img_px))
      data = np.vstack([full_img_data[idx,...].reshape(self.num_patches, patch_edge_size,
        patch_edge_size) for idx in range(num_img)])
    else:
      data = full_img_data
      self.num_patches = 0
    return data

def load_vanHateren(
  data_dir,
  normalize_imgs=False,
  whiten_imgs=True,
  patch_edge_size=None,
  rand_state=np.random.RandomState()):

  ## Training set
  img_filename = data_dir+os.sep+"images_curated.h5"
  vh_data = vanHateren(
    img_filename,
    patch_edge_size,
    rand_state=rand_state)
  images = Dataset(vh_data.images, None, None, normalize=normalize_imgs,
    rand_state=rand_state)
  setattr(images, "num_patches", vh_data.num_patches)
  return {"train":images}
