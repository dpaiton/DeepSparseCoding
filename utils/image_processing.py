import numpy as np
import scipy.ndimage

"""
Extract patches from image dataset.
Outputs:
  patches [np.ndarray] of patches
Inputs:
  images [np.ndarray] of shape [num_images, img_height, img_width]
  out_shape [tuple or list] containing the 2-d output shape
    [num_patches, patch_size] where patch_size has an even sqrt
    [num_patches, patch_edge_size, patch_edge_size]
  overlapping [bool] specify if the patches are evenly tiled or randomly drawn
  var_thresh [float] acceptance threshold for patch pixel variance. If it is
    below threshold then reject the patch.
"""
def extract_patches(images, out_shape, overlapping=True, var_thresh=0,
  rand_state=np.random.RandomState()):
  images = reshape_data(images, flatten=False)[0]
  num_im, im_sizey, im_sizex = images.shape
  if len(out_shape) == 2:
    (num_patches, patch_size) = out_shape
    assert np.floor(np.sqrt(patch_size)) == np.ceil(np.sqrt(patch_size)), (
      "Patch size must have an even square root.")
    patch_edge_size = np.int32(np.sqrt(patch_size))
  elif len(out_shape) == 3:
    (num_patches, patch_y_size, patch_x_size) = out_shape
    assert patch_y_size == patch_x_size, ("Patches must be square.")
    patch_edge_size = patch_y_size
    patch_size = patch_edge_size**2
  else:
    assert False, ("out_shape must have len 2 or 3.")
  if (patch_edge_size <= 0 or patch_edge_size == im_sizey):
    if num_patches < num_im:
      im_keep_idx = rand_state.choice(images.shape[0], num_patches,
        replace=False)
      return images[im_keep_idx, ...]
    elif num_patches == num_im:
      return images
    else:
      assert False, (
        "The number of requested %g pixel patches must be less than or equal "
        +"to %g"%(patch_size, num_im))
  if overlapping:
    patches = np.zeros((num_patches, patch_size))
    i = 0
    while i < num_patches:
      image = images[rand_state.randint(num_im), ...]
      row = rand_state.randint(im_sizey - patch_edge_size)
      col = rand_state.randint(im_sizex - patch_edge_size)
      patch = image[row:row+patch_edge_size, col:col+patch_edge_size]
      if np.var(patch) > var_thresh:
        patches[i, :] = np.reshape(patch, patch_size)
        i = i+1
  else:
    num_available_patches = num_im * np.floor(im_sizex/patch_edge_size)**2
    assert num_patches <= num_available_patches, (
      "The number of requested patches (%g) must be less than or equal to %g"%(
      num_patches, num_available_patches))
    if im_sizex % patch_edge_size != 0: # crop columns
      crop_x = im_sizex % patch_edge_size
      crop_edge = np.int32(np.floor(crop_x/2.0))
      images = images[:, crop_edge:im_sizex-crop_edge, :]
      im_sizex = images.shape[1]
    if im_sizey % patch_edge_size != 0: # crop rows
      crop_y = im_sizey % patch_edge_size
      crop_edge = np.int32(np.floor(crop_y/2.0))
      images = images[:, :, crop_edge:im_sizey-crop_edge]
      im_sizey = images.shape[2]
    # Tile column-wise, then row-wise
    patches = np.asarray(np.split(images, im_sizex/patch_edge_size, 2))
    patches = np.asarray(np.split(patches, im_sizey/patch_edge_size, 2))
    patches = np.transpose(np.reshape(np.transpose(patches, axes=(3,4,0,1,2)),
      (patch_edge_size, patch_edge_size, -1)), axes=(2,0,1))
    patches = patches[(np.var(patches, axis=(1,2)) > var_thresh)]
    if patches.shape[0] < num_patches:
      assert False, (
        "out_shape requres too many patches (%g); maximum available is %g."%(
        num_patches, patches.shape[0]))
    else:
      patch_keep_idx = rand_state.choice(patches.shape[0], num_patches,
        replace=False)
      patches = patches[patch_keep_idx, ...]
  if len(out_shape) == 2:
    return patches
  else:
    return patches.reshape(num_patches, patch_edge_size, patch_edge_size)

"""
Downsample data
currently uses scipy.ndimage.zoom
I would like to implement a (windowed?) gaussian convolution instead
"""
def downsample_data(data, factor, order):
  return scipy.ndimage.interpolation.zoom(data, factor, order=order)

"""
Helper function to reshape input data for processing and return data shape
Outputs:
  tuple containing:
  data [np.ndarray] data with new shape
    (num_examples, num_rows, num_cols) if flatten==False
    (num_examples, num_elements) if flatten==True
  orig_shape [tuple of int32] original shape of the input data
  num_examples [int32] number of data examples
  num_rows [int32] number of data rows (sqrt of num elements)
  num_cols [int32] number of data cols (sqrt of num elements)
Inputs: 
  data [np.ndarray] unnormalized data of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k
  flatten [bool] if True, 
"""
def reshape_data(data, flatten=False):
  orig_shape = data.shape
  orig_ndim = data.ndim
  if orig_ndim == 1:
    num_examples = 1
    num_elements = data.shape[0]
    sqrt_num_elements = np.sqrt(num_elements)
    assert np.floor(sqrt_num_elements) == np.ceil(sqrt_num_elements), (
      "Data length must have an even square root.")
    num_rows = np.int32(np.floor(sqrt_num_elements))
    num_cols = num_rows
    if flatten:
      data = data[np.newaxis, ...]
    else:
      data = data.reshape((num_rows, num_cols))[np.newaxis, ...]
  elif orig_ndim == 2:
    (num_examples, num_elements) = data.shape
    sqrt_num_elements = np.sqrt(num_elements)
    num_rows = np.int32(np.floor(sqrt_num_elements))
    num_cols = (num_rows
      + np.int32(np.ceil(sqrt_num_elements)-np.floor(sqrt_num_elements)))
    if not flatten:
      assert np.floor(sqrt_num_elements) == np.ceil(sqrt_num_elements), (
        "Data length must have an even square root.")
      data = data.reshape((num_examples, num_rows, num_cols))
  elif orig_ndim == 3:
    (num_examples, num_rows, num_cols) = data.shape
    assert num_rows == num_cols, ("Data points must be square.")
    if flatten:
      data = data.reshape((num_examples, num_rows * num_cols))
  else:
    assert False, ("Data must have 1, 2, or 3 dimensions.")
  return (data, orig_shape, num_examples, num_rows, num_cols)

"""
Normalize data by dividing by abs(max(data))
Outputs:
  norm_data: [np.ndarray] data normalized so that 0 is midlevel grey
Inputs:
  data: [np.ndarray] data to be normalized
"""
def normalize_data_with_max(data):
  if np.max(np.abs(data)) > 0:
    norm_data = (data / np.max(np.abs(data))).squeeze()
  else:
    norm_data = data.squeeze()
  return norm_data

"""
Subtract individual example mean from data
Outputs:
  data [np.ndarray] centered data
Inputs:
  data [np.ndarray] unnormalized data of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k
"""
def center_data(data):
  data = data[np.newaxis, ...] if data.ndim == 1 else data
  for idx in range(data.shape[0]):
    data[idx, ...] -= np.mean(data[idx, ...])
  return data.squeeze()

"""
Standardize data to have zero mean and unit variance (z-score)
Outputs:
  data [np.ndarray] normalized data
Inputs:
  data [np.ndarray] unnormalized data of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k
TODO:
  look into tf.image.per_image_standardization()
"""
def standardize_data(data):
  data = data[np.newaxis, ...] if data.ndim == 1 else data
  for idx in range(data.shape[0]):
    data[idx, ...] -= np.mean(data[idx, ...])
    data[idx, ...] = data[idx, ...] / np.std(data[idx, ...])
  return data.squeeze()

"""
Whiten data
Outputs:
  whitened_data
Inputs:
  data: [np.ndarray] of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k
  method: [str] method to use, can be {FT, PCA}
  num_dim: [int] specifies the number of PCs to use for PCA method
"""
def whiten_data(data, method="FT", num_dim=-1):
  if method == "FT":
    (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
      flatten=False) # Need spatial dim for 2d-Fourier transform
    data -= data.mean(axis=(1,2))[:,None,None]
    dataFT = np.fft.fftshift(np.fft.fft2(data, axes=(1, 2)), axes=(1, 2))
    nyq = np.int32(np.floor(num_rows/2))
    freqs = np.linspace(-nyq, nyq-1, num=num_rows)
    fspace = np.meshgrid(freqs, freqs)
    rho = np.sqrt(np.square(fspace[0]) + np.square(fspace[1]))
    lpf = np.exp(-0.5 * np.square(rho / (0.7 * nyq)))
    filtf = np.multiply(rho, lpf)
    dataFT_wht = np.multiply(dataFT, filtf[None, :])
    data_wht = np.real(np.fft.ifft2(np.fft.ifftshift(dataFT_wht, axes=(1, 2)),
      axes=(1, 2)))
    data_wht = reshape_data(data_wht, flatten=True)[0]
  elif method == "PCA":
    (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
      flatten=True)
    data -= data.mean(axis=(1))[:,None]
    Cov = np.cov(data.T) # Covariace matrix
    U, S, V = np.linalg.svd(Cov) # SVD decomposition
    isqrtS = np.diag(1 / np.sqrt(S)) # Inverse sqrt of S
    data_wht = np.dot(data, np.dot(np.dot(U, isqrtS), V))
  else:
    assert False, ("whitening method must be 'FT' or 'PCA'")
  return data_wht

def pca_reduction(data, num_pcs=-1):
  (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
    flatten=True)
  data_mean = data.mean(axis=(1))[:,None]
  data -= data_mean
  Cov = np.cov(data.T) # Covariace matrix
  U, S, V = np.linalg.svd(Cov) # SVD decomposition
  diagS = np.diag(S)
  if num_pcs <= 0:
    n = num_rows
  else:
    n = num_pcs
  data_reduc = np.dot(data, np.dot(np.dot(U[:, :n], diagS[:n, :n]), V[:n, :]))
  if orig_shape != data.shape:
    data_reduc = reshape_data(data_reduc, flatten=False)[0]
  return data_reduc

"""
Compute Fourier power spectrum for input data
Outputs:
  power_spec: [np.ndarray] Fourier power spectrum
Inputs:
  data: [np.ndarray] of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k (k must have even sqrt)
"""
def compute_power_spectrum(data):
  (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
    flatten=False)
  data = standardize_data(data)
  dataFT = np.fft.fftshift(np.fft.fft2(data, axes=(1, 2)), axes=(1, 2))
  power_spec = np.multiply(dataFT, np.conjugate(dataFT)).real
  return power_spec

"""
Compute phase average of power spectrum
Only works for greyscale imagery
Outputs:
  phase_avg: [list of np.ndarray] phase averaged power spectrum
    each element in the list corresponds to a data point
Inputs:
  data: [np.ndarray] of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k (k must have even sqrt)
"""
def phase_avg_pow_spec(data):
  (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
    flatten=False)
  power_spec = compute_power_spectrum(data)
  dims = power_spec[0].shape
  nyq = np.int32(np.floor(np.array(dims)/2.0))
  freqs = [np.linspace(-nyq[i], nyq[i]-1, num=dims[i])
    for i in range(len(dims))]
  fspace = np.meshgrid(freqs[0], freqs[1], indexing='ij')
  rho = np.round(np.sqrt(np.square(fspace[0]) + np.square(fspace[1])))
  phase_avg = np.zeros((num_examples, nyq[0]))
  for data_idx in range(num_examples):
    for rad in range(nyq[0]):
      if not np.isnan(np.mean(power_spec[data_idx][rho == rad])):
        phase_avg[data_idx, rad] = np.mean(power_spec[data_idx][rho == rad])
  return phase_avg
