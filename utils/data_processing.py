import numpy as np
import scipy.signal
import scipy.stats
import scipy.ndimage

def reshape_data(data, flatten=None, out_shape=None):
  """
  Helper function to reshape input data for processing and return data shape
  Inputs:
    data: [np.ndarray] data of shape:
      (n, i, j, k) - n data points, each of shape (i,j,k), with k specifying num_channels
      (n, l) - n data points, each of shape l (flattened)
      (i, j, k) - single datapoint of of shape (i,j, k), a singleton dimension will be added to the output
      (l) - single data point of shape l, assumes 1 color channel
    flatten: [bool or None] specify the shape of the output
      If None, do not reshape data, but add num_examples dimension if necessary
      If True, return ravelled data of shape (num_examples, num_elements)
      If False, return unravelled data, of shape out_shape or of shape (num_examples, sqrt(l), sqrt(l), 1)
      If data is flat and flatten==True, or !flat and flatten==False, then None condition will apply
    out_shape: (optional) [list or tuple] containing the desired output shape if flatten == False
      should specify [num_rows, num_cols, num_channels]
      If this is not given, the default is to assume a square image and num_channels = 1
  Outputs:
    tuple containing:
    data: [np.ndarray] data with new shape
      (num_examples, num_rows, num_cols, num_channels) if flatten==False
      (num_examples, num_elements) if flatten==True
    orig_shape: [tuple of int32] original shape of the input data
      Note that if the input data did not have a num_examples dim,
      then the output data will and orig_shape will include a singleton dimension for num_examples
    num_examples: [int32] number of data examples
    num_rows: [int32] number of data rows
    num_cols: [int32] number of data cols
    num_channels: [int32] number of data channels
  """
  orig_shape = list(data.shape)
  orig_ndim = data.ndim
  if orig_ndim == 1: # single datapoint
    num_examples = 1
    num_channels = 1
    num_elements = data.shape[0]
    if flatten is None or flatten == True:
      num_rows = num_elements
      num_cols = 1
      data = data[None, ...]
      orig_shape = [1]+orig_shape
    else:
      sqrt_num_elements = np.sqrt(num_elements)
      assert np.floor(sqrt_num_elements) == np.ceil(sqrt_num_elements), (
        "Data length must have an even square root.")
      num_rows = int(sqrt_num_elements)
      num_cols = num_rows
      data = data.reshape((num_rows, num_cols))[None, ..., None]
      orig_shape = [1]+orig_shape+[1]
  elif orig_ndim == 2: # already flattened
    (num_examples, num_elements) = data.shape
    if flatten is None or flatten == True:
      num_rows = num_elements
      num_cols = 1
      num_channels = 1
    else:
      if out_shape is not None:
        num_rows, num_cols, num_channels = out_shape
        data = data.reshape((num_examples, num_rows, num_cols, num_channels))
      else:
        sqrt_num_elements = np.sqrt(num_elements)
        assert np.floor(sqrt_num_elements) == np.ceil(sqrt_num_elements), (
          "Data length must have an even square root when not specifying out_shape.")
        num_rows = int(sqrt_num_elements)
        num_cols = num_rows
        num_channels = 1
        data = data.reshape((num_examples, num_rows, num_cols, num_channels))
  elif orig_ndim == 3: # single data point
    num_examples = 1
    num_rows, num_cols, num_channels = data.shape
    if flatten == True:
      data = data.reshape((num_examples, num_rows * num_cols * num_channels))
    elif flatten is None or flatten == False:
      data = data[None, ...]
      orig_shape = [1]+orig_shape
    else:
        assert False, ("flatten argument must be True, False, or None")
  elif orig_ndim == 4: # not flat
    num_examples, num_rows, num_cols, num_channels = data.shape
    if flatten == True:
      data = data.reshape((num_examples, num_rows*num_cols*num_channels))
  else:
    assert False, ("Data must have 1, 2, 3, or 4 dimensions.")
  return (data, tuple(orig_shape), num_examples, num_rows, num_cols, num_channels)


def hilbert_amplitude(weights, padding=None):
  """
  Compute Hilbert amplitude envelope of weight matrix
  Inputs:
    weights: [np.ndarray] of shape [num_inputs, num_outputs]
      num_inputs must have an even square root
    padding: [int] specifying how much 0-padding to use for FFT
      default is the closest power of 2 of sqrt(num_inputs)
  Outputs:
    env: [np.ndarray] of shape [num_outputs, num_inputs]
      Hilbert envelope
    bff_filt: [np.ndarray] of shape [num_outputs, padded_num_inputs]
      Filtered Fourier transform of basis function
    hil_filt: [np.ndarray] of shape [num_outputs, sqrt(num_inputs), sqrt(num_inputs)]
      Hilbert filter to be applied in Fourier space
    bffs: [np.ndarray] of shape [num_outputs, padded_num_inputs, padded_num_inputs]
      Fourier transform of input weights
  """
  cart2pol = lambda x,y: (np.arctan2(y,x), np.hypot(x, y))
  num_inputs, num_outputs = weights.shape
  assert np.sqrt(num_inputs) == np.floor(np.sqrt(num_inputs)), (
    "weights.shape[0] must have an even square root.")
  patch_edge_size = int(np.sqrt(num_inputs))
  if padding is None or padding <= patch_edge_size:
    # Amount of zero padding for fft2 (closest power of 2)
    N = np.int(2**(np.ceil(np.log2(patch_edge_size))))
  else:
    N = np.int(padding)
  # Analytic signal envelope for weights
  # (Hilbet transform of each basis function)
  env = np.zeros((num_outputs, num_inputs), dtype=complex)
  # Fourier transform of weights
  bffs = np.zeros((num_outputs, N, N), dtype=complex)
  # Filtered Fourier transform of weights
  bff_filt = np.zeros((num_outputs, N**2), dtype=complex)
  # Hilbert filters
  hil_filt = np.zeros((num_outputs, N, N))
  # Grid for creating filter
  f = (2/N) * np.pi * np.arange(-N/2.0, N/2.0)
  (fx, fy) = np.meshgrid(f, f)
  (theta, r) = cart2pol(fx, fy)
  for neuron_idx in range(num_outputs):
    # Grab single basis function, reshape to a square image
    bf = weights[:, neuron_idx].reshape(patch_edge_size, patch_edge_size)
    # Convert basis function into DC-centered Fourier domain
    bff = np.fft.fftshift(np.fft.fft2(bf-np.mean(bf), [N, N]))
    bffs[neuron_idx, ...] = bff
    # Find indices of the peak amplitude
    max_ys = np.abs(bff).argmax(axis=0) # Returns row index for each col
    max_x = np.argmax(np.abs(bff).max(axis=0))
    # Convert peak amplitude location into angle in freq domain
    fx_ang = f[max_x]
    fy_ang = f[max_ys[max_x]]
    theta_max = np.arctan2(fy_ang, fx_ang)
    # Define the half-plane with respect to the maximum
    ang_diff = np.abs(theta-theta_max)
    idx = (ang_diff>np.pi).nonzero()
    ang_diff[idx] = 2.0 * np.pi - ang_diff[idx]
    hil_filt[neuron_idx, ...] = (ang_diff < np.pi/2.0).astype(int)
    # Create analytic signal from the inverse FT of the half-plane filtered bf
    abf = np.fft.ifft2(np.fft.fftshift(hil_filt[neuron_idx, ...]*bff))
    env[neuron_idx, ...] = abf[0:patch_edge_size, 0:patch_edge_size].reshape(num_inputs)
    bff_filt[neuron_idx, ...] = (hil_filt[neuron_idx, ...]*bff).reshape(N**2)
  return (env, bff_filt, hil_filt, bffs)

def get_dictionary_stats(weights, padding=None, num_gauss_fits=20, gauss_thresh=0.2):
  """
  Compute summary statistics on dictionary elements using Hilbert amplitude envelope
  Inputs:
    weights: [np.ndarray] of shape [num_inputs, num_outputs]
    padding: [int] total image size to pad out to in the FFT computation
    num_gauss_fits: [int] total number of attempts to make when fitting the BFs
    gauss_thresh: All probability values below gauss_thresh*mean(gauss_fit) will be
      considered outliers for repeated fits
  Outputs:
    The function output is a dictionary containing the keys for each type of analysis
    Each key dereferences a list of len num_outputs (i.e. one entry for each weight vector)
    The keys and their list entries are as follows:
      basis_functions: [np.ndarray] of shape [patch_edge_size, patch_edge_size]
      envelopes: [np.ndarray] of shape [N, N], where N is the amount of padding
        for the hilbert_amplitude function
      envelope_centers: [tuples of ints] indicating the (y, x) position of the
        center of the Hilbert envelope
      gauss_fits: [list of np.ndarrays] containing (gaussian_fit, grid) where gaussian_fit
        is returned from get_gauss_fit and specifies the 2D Gaussian PDF fit to the Hilbert
        envelope and grid is a tuple containing (y,x) points with which the Gaussian PDF
        can be plotted
      gauss_centers: [list of ints] containing the (y,x) position of the center of
        the Gaussian fit
      gauss_orientations: [list of np.ndarrays] containing the (eigenvalues, eigenvectors) of
        the covariance matrix for the Gaussian fit of the Hilbert amplitude envelope. They are
        both sorted according to the highest to lowest Eigenvalue.
      fourier_centers: [list of ints] containing the (y,x) position of the center (max) of
        the Fourier amplitude map
      num_inputs: [int] dim[0] of input weights
      num_outputs: [int] dim[1] of input weights
      patch_edge_size: [int] int(floor(sqrt(num_inputs)))
      areas: [list of floats] area of enclosed ellipse
      spatial_frequncies: [list of floats] dominant spatial frequency for basis function
  """
  envelope, bff_filt, hil_filter, bffs = hilbert_amplitude(weights, padding)
  num_inputs, num_outputs = weights.shape
  patch_edge_size = np.int(np.floor(np.sqrt(num_inputs)))
  basis_funcs = [None]*num_outputs
  envelopes = [None]*num_outputs
  gauss_fits = [None]*num_outputs
  gauss_centers = [None]*num_outputs
  gauss_orientations = [None]*num_outputs
  envelope_centers = [None]*num_outputs
  fourier_centers = [None]*num_outputs
  fourier_maps = [None]*num_outputs
  spatial_frequencies = [None]*num_outputs
  areas = [None]*num_outputs
  for bf_idx in range(num_outputs):
    # Reformatted individual basis function
    basis_funcs[bf_idx] = np.squeeze(reshape_data(weights.T[bf_idx,...],
      flatten=False)[0])
    # Reformatted individual envelope filter
    envelopes[bf_idx] = np.squeeze(reshape_data(np.abs(envelope[bf_idx,...]),
      flatten=False)[0])
    # Basis function center
    max_ys = envelopes[bf_idx].argmax(axis=0) # Returns row index for each col
    max_x = np.argmax(envelopes[bf_idx].max(axis=0))
    y_cen = max_ys[max_x]
    x_cen = max_x
    envelope_centers[bf_idx] = (y_cen, x_cen)
    # Gaussian fit to Hilbet amplitude envelope
    gauss_fit, grid, gauss_mean, gauss_cov = get_gauss_fit(envelopes[bf_idx],
      num_gauss_fits, gauss_thresh)
    gauss_fits[bf_idx] = (gauss_fit, grid)
    gauss_centers[bf_idx] = gauss_mean
    evals, evecs = np.linalg.eigh(gauss_cov)
    sort_indices = np.argsort(evals)[::-1]
    gauss_orientations[bf_idx] = (evals[sort_indices], evecs[:,sort_indices])
    # Fourier function center, spatial frequency, orientation
    fourier_map = np.sqrt(np.real(bffs[bf_idx, ...])**2+np.imag(bffs[bf_idx, ...])**2)
    fourier_maps[bf_idx] = fourier_map
    N = fourier_map.shape[0]
    center_freq = int(np.floor(N/2))
    fourier_map[center_freq, center_freq] = 0 # remove DC component
    max_fys = fourier_map.argmax(axis=0)
    max_fx = np.argmax(fourier_map.max(axis=0))
    fy_cen = (max_fys[max_fx] - (N/2)) * (patch_edge_size/N)
    fx_cen = (max_fx - (N/2)) * (patch_edge_size/N)
    fourier_centers[bf_idx] = [fy_cen, fx_cen]
    spatial_frequencies[bf_idx] = np.sqrt(fy_cen**2 + fx_cen**2)
    areas[bf_idx] = np.pi * np.prod(evals)
  output = {"basis_functions":basis_funcs, "envelopes":envelopes, "gauss_fits":gauss_fits,
    "gauss_centers":gauss_centers, "gauss_orientations":gauss_orientations, "areas":areas,
    "fourier_centers":fourier_centers, "fourier_maps":fourier_maps, "num_inputs":num_inputs,
    "spatial_frequencies":spatial_frequencies, "envelope_centers":envelope_centers,
    "num_outputs":num_outputs, "patch_edge_size":patch_edge_size}
  return output

def generate_gaussian(shape, mean, cov):
  """
  Generate a Gaussian PDF from given mean & cov
  Inputs:
    shape: [tuple] specifying (num_rows, num_cols)
    mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
    cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
  Outputs:
    tuple containing (Gaussian PDF, grid_points used to generate PDF)
      grid_points are specified as a tuple of (y,x) points
  """
  (y_size, x_size) = shape
  y = np.linspace(0, y_size, np.int32(np.floor(y_size)))
  x = np.linspace(0, x_size, np.int32(np.floor(x_size)))
  y, x = np.meshgrid(y, x)
  pos = np.empty(x.shape + (2,)) #x.shape == y.shape
  pos[:, :, 0] = y; pos[:, :, 1] = x
  gauss = scipy.stats.multivariate_normal(mean, cov)
  return (gauss.pdf(pos), (y,x))

def gaussian_fit(pyx):
  """
  Compute the expected mean & covariance matrix for a 2-D gaussian fit of input distribution
  Inputs:
    pyx: [np.ndarray] of shape [num_rows, num_cols] that indicates the probability function to fit
  Outputs:
    mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
    cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
  """
  assert pyx.ndim == 2, (
    "Input must have 2 dimensions specifying [num_rows, num_cols]")
  mean = np.zeros((1,2), dtype=np.float32) # [mu_y, mu_x]
  for idx in np.ndindex(pyx.shape): # [y, x] ticks columns (x) first, then rows (y)
    mean += np.asarray([pyx[idx]*idx[0], pyx[idx]*idx[1]])[None,:]
  cov = np.zeros((2,2), dtype=np.float32)
  for idx in np.ndindex(pyx.shape): # ticks columns first, then rows
    cov += np.dot((idx-mean).T, (idx-mean))*pyx[idx] # typically an outer-product
  return (np.squeeze(mean), cov)

def get_gauss_fit(prob_map, num_attempts=1, perc_mean=0.33):
  """
  Returns a gaussian fit for a given probability map
  Fitting is done via robust regression, where a fit is
  continuously refined by deleting outliers num_attempts times
  Inputs:
    prob_map: 2-D probability map to be fit
    num_attempts: Number of times to fit & remove outliers
    perc_mean: All probability values below perc_mean*mean(gauss_fit) will be
      considered outliers for repeated attempts
  Outputs:
    gauss_fit: [np.ndarray] specifying the 2-D Gaussian PDF
    grid: [tuple] containing (y,x) points with which the Gaussian PDF can be plotted
    gauss_mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
    gauss_cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
  """
  assert prob_map.ndim==2, (
    "get_gauss_fit: Input prob_map must have 2 dimension specifying [num_rows, num_cols")
  if num_attempts < 1:
    num_attempts = 1
  orig_prob_map = prob_map.copy()
  gauss_success = False
  while not gauss_success:
    prob_map = orig_prob_map.copy()
    try:
      for i in range(num_attempts):
        map_min = np.min(prob_map)
        prob_map -= map_min
        map_sum = np.sum(prob_map)
        if map_sum != 1.0:
          prob_map /= map_sum
        gauss_mean, gauss_cov = gaussian_fit(prob_map)
        gauss_fit, grid = generate_gaussian(prob_map.shape, gauss_mean, gauss_cov)
        gauss_fit = (gauss_fit * map_sum) + map_min
        if i < num_attempts-1:
          gauss_mask = gauss_fit.copy().T
          gauss_mask[np.where(gauss_mask<perc_mean*np.mean(gauss_mask))] = 0
          gauss_mask[np.where(gauss_mask>0)] = 1
          prob_map *= gauss_mask
      gauss_success = True
    except np.linalg.LinAlgError: # Usually means cov matrix is singular
      print("get_gauss_fit: Failed to fit Gaussian at attempt ",i,", trying again."+
        "\n  To avoid this try decreasing perc_mean.")
      num_attempts = i-1
      if num_attempts <= 0:
        assert False, ("get_gauss_fit: np.linalg.LinAlgError - Unable to fit gaussian.")
  return (gauss_fit, grid, gauss_mean, gauss_cov)

def extract_overlapping_patches(images, out_shape, var_thresh=0,
  rand_state=np.random.RandomState()):
  """
  Extract randomly selected, overlapping patches from image dataset.
  Inputs:
    images [np.ndarray] of shape [num_images, im_height, im_width, im_chan]
    out_shape [tuple or list] containing the output shape
      [num_patches, patch_height, patch_width, patch_chan]
      patch_chan must be the same as im_chan
    var_thresh [float] acceptance threshold for patch pixel variance. If it is
      below threshold then reject the patch.
    rand_state [np.random.RandomState()]
  Outputs:
    patches [np.ndarray] of patches of shape out_shape
  TODO:
    Allow non-random overlapping patches (e.g. strided convolution patches)
  """
  num_im, im_height, im_width, im_chan = images.shape
  num_patches, patch_height, patch_width, patch_chan = out_shape
  assert im_chan == patch_chan, (
    "out_shape must specify the same number of channels as the input images")
  patch_size = out_shape[1:]
  patches = np.zeros(out_shape, dtype=np.float32)
  i = 0
  while i < num_patches:
    example = rand_state.randint(num_im)
    row = rand_state.randint(im_height - patch_height)
    col = rand_state.randint(im_width - patch_width)
    patch = images[example, row:row+patch_height, col:col+patch_width, ...]
    if np.var(patch) > var_thresh:
      patches[i, :] = np.reshape(patch, patch_size)
      i = i+1
  return patches

def extract_random_tiled_patches(images, out_shape, var_thresh=0,
  rand_state=np.random.RandomState()):
  """
  Extract randomly selected non-overlapping patches from image dataset.
  Inputs:
    images [np.ndarray] of shape [num_images, im_height, im_width, im_chan]
    out_shape [tuple or list] containing the output shape
      [num_patches, patch_height, patch_width, patch_chan]
      patch_chan must be the same as im_chan
    var_thresh [float] acceptance threshold for patch pixel variance. If it is
      below threshold then reject the patch.
    rand_state [np.random.RandomState()]
  Outputs:
    patches [np.ndarray] of patches of shape out_shape
  """
  num_im, im_height, im_width, im_chan = images.shape
  num_patches, patch_height, patch_width, patch_chan = out_shape
  num_row_patches = num_im * np.floor(im_height / patch_height)
  num_col_patches = num_im * np.floor(im_width / patch_width)
  num_available_patches = int(num_row_patches * num_col_patches)
  assert num_patches <= num_available_patches, (
    "The number of requested patches (%g) must be less than or equal to %g"%(
      num_patches, num_available_patches))
  if im_height % patch_height != 0: # crop rows
    crop_rows = im_height % patch_height
    crop_edge = np.int32(np.floor(crop_rows/2.0))
    images = images[:, crop_edge:im_height-crop_edge, :, :]
    im_height = images.shape[1]
  if im_width % patch_width != 0: # crop columns
    crop_cols = im_width % patch_width
    crop_edge = np.int32(np.floor(crop_cols/2.0))
    images = images[:, :, crop_edge:im_width-crop_edge, :]
    im_width = images.shape[2]
  import IPython; IPython.embed()
  # Tile column-wise, then row-wise
  patches = np.asarray(np.split(images, im_width/patch_width, axis=2))
  # patches.shape = [im_width/patch_width, num_im, im_height, patch_height, patch_chan]
  patches = np.asarray(np.split(patches, im_height/patch_height, axis=2))
  # patches.shape = [im_height/patch_height, im_width/patch_width, num_im,
  #  patch_height, patch_width, patch_chan]
  patches = np.transpose(patches, axes=(3,4,5,0,1,2))
  # patches.shape = [patch_height, patch_width, patch_chan, im_height/patch_height,
  #   im_width/patch_width, num_im]
  patches = np.reshape(patches, (patch_height, patch_width, patch_chan, -1))
  # patches.shape = [patch_height, patch_width, patch_chan, num_patches]
  patches = np.transpose(patches, axes=(3,0,1,2))
  # patches.shape = [num_patches, patch_height, patch_width, patch_chan]
  patches = patches[(np.var(patches, axis=(1,2)) > var_thresh)]
  assert patches.shape[0] >= num_patches, (
    "out_shape (%g) requres too many patches; maximum available is %g."%(
    num_patches, patches.shape[0]))
  patch_keep_idx = rand_state.choice(patches.shape[0], num_patches, replace=False)
  patch_keep_idx = np.arange(num_patches)
  patches = patches[patch_keep_idx, ...]
  return patches

def extract_patches_from_single_image(image, out_shape):
  """
  Extract patches from a single image
  Inputs:
    image [np.ndarray] of shape [im_height, im_width, im_chan]
    out_shape [tuple or list] containing the output shape
      [patch_height, patch_width, patch_chan]
      patch_chan must be the same as im_chan
  Outputs:
    patches [np.ndarray] of patches of shape [num_patches, patch_height, patch_width, patch_chan]
  """
  assert image.ndim == 3, ("input must have 3 ndim")
  im_height, im_width, im_chan = image.shape
  patch_height, patch_width, patch_chan = out_shape
  assert im_chan == patch_chan, ("image and out_shape must specify the same number of channels")
  assert im_height % patch_height == 0, ("image height (%g) must equal patch height (%g)"%(
    im_height, patch_height))
  assert im_width % patch_width == 0, ("image width (%g) must equal patch width (%g)"%(
    im_width, patch_width))
  num_row_patches = np.floor(im_height / patch_height)
  num_col_patches = np.floor(im_width / patch_width)
  num_patches = int(num_row_patches * num_col_patches)
  patches = np.zeros((num_patches, patch_height, patch_width, patch_chan))
  row_id = 0
  col_id = 0
  for patch_idx in range(num_patches):
    patches[patch_idx, ...] = image[row_id:row_id+patch_height, col_id:col_id+patch_width, :]
    row_id += patch_height
    if row_id >= im_height:
      row_id = 0
      col_id += patch_width
    if col_id >= im_width:
      col_id = 0
  return patches

def extract_tiled_patches(images, out_shape):
  """
  Extract tiled patches from image dataset.
  Inputs:
    image [np.ndarray] of shape [num_im, im_height, im_width, im_chan] or
      [im_height, im_width, im_chan] if only using one image
    out_shape [tuple or list] containing the output shape
      [patch_height, patch_width, patch_chan]
      patch_chan must be the same as im_chan
      note that out_shape doesn't specify num_patches, it extracts all patches from the given images
  Outputs:
    patches [np.ndarray] of patches of shape out_shape
  """
  if images.ndim == 3: # single image of dim [im_height, im_width, im_chan]
      return extract_patches_from_single_image(images, out_shape)
  num_im, im_height, im_width, im_chan = images.shape
  patch_height, patch_width, patch_chan = out_shape
  assert im_chan == patch_chan, ("image and out_shape must specify the same number of channels")
  assert im_height % patch_height == 0, ("image height (%g) must equal patch height (%g)"%(
    im_height, patch_height))
  assert im_width % patch_width == 0, ("image width (%g) must equal patch width (%g)"%(
    im_width, patch_width))
  num_row_patches = np.floor(im_height / patch_height)
  num_col_patches = np.floor(im_width / patch_width)
  num_patches_per_im = int(num_row_patches * num_col_patches)
  tot_num_patches =  int(num_patches_per_im * num_im)
  patch_list = [None,]*num_im
  patch_id = 0
  for im_id in range(num_im):
    patch_list[patch_id] = extract_patches_from_single_image(images[im_id, ...], out_shape)
    patch_id += 1
  patches = np.stack(patch_list)
  # patches.shape = [num_im, num_patches_per_im, patch_height, patch_width, patch_chan]
  patches = np.transpose(patches, axes=(2,3,4,0,1))
  # patches.shape = [patch_height, patch_width, patch_chan, num_im, num_patches_per_im]
  patches = np.reshape(patches, (patch_height, patch_width, patch_chan, -1))
  # patches.shape = [patch_height, patch_width, patch_chan, num_patches]
  patches = np.transpose(patches, axes=(3,0,1,2))
  # patches.shape = [num_patches, patch_height, patch_width, patch_chan]
  return patches

def extract_patches(images, out_shape, overlapping=False, randomize=False, var_thresh=0,
  rand_state=np.random.RandomState()):
  """
  Extract patches from image dataset.
  Inputs:
    images [np.ndarray] of shape [num_images, im_height, im_width, im_chan]
      or [im_height, im_width, im_chan] for a single image
    out_shape [tuple or list] containing the output shape
      [num_patches, patch_height, patch_width, patch_chan]
      patch_chan must be the same as im_chan
    overlapping [bool] specify if the patches are evenly tiled or randomly drawn
    randomize [bool] specify if the patches are drawn randomly (must be True for overlapping)
    var_thresh [float] acceptance threshold for patch pixel variance. If it is
      below threshold then reject the patch.
    rand_state [np.random.RandomState()] for reproducibility
  Outputs:
    patches [np.ndarray] of patches
  """
  if images.ndim == 3: # single image
    images = images[None,...]
  num_im, im_height, im_width, im_chan = images.shape
  num_patches, patch_height, patch_width, patch_chan = out_shape
  if patch_height == im_height and patch_width == im_width:
    if num_patches < num_im:
      im_keep_idx = rand_state.choice(images.shape[0], num_patches, replace=False)
      return images[im_keep_idx, ...]
    elif num_patches == num_im:
      return images
    else:
      assert False, (
        "The number of requested patches (%g) must be less than or equal to %g."%(
        num_patches, num_im))
  if overlapping:
    patches = extract_overlapping_patches(images, out_shape, var_thresh, rand_state)
  else:
    if randomize:
      patches = extract_random_tiled_patches(images, out_shape, var_thresh, rand_state)
    else:
      patches = extract_tiled_patches(images, out_shape[1:])
  return patches

def patches_to_single_image(patches, im_shape):
  """
  Convert patches input into a single ouput
  Inputs:
    patches [np.ndarray] of shape [num_patches, patch_height, patch_width, patch_chan]
    im_shape [list or tuple] containing the image shape
      [im_height, im_width, im_chan]
      im_chan must equal patch_chan
  """
  num_patches, patch_height, patch_width, patch_chan = patches.shape
  im_height, im_width, im_chan = im_shape
  assert im_chan == patch_chan, ("specified im_shape must have same number of channels as patches.")
  im = np.zeros((im_height, im_width, im_chan))
  row_id = 0
  col_id = 0
  for patch_idx in range(num_patches):
    im[row_id:row_id+patch_height, col_id:col_id+patch_width] = patches[patch_idx,...]
    row_id += patch_height
    if row_id >= im_height:
      row_id = 0
      col_id += patch_width
    if col_id >= im_width:
      col_id = 0
  return im

def patches_to_image(patches, im_shape):
  """
  Reassemble patches created from extract_tiled_patches() into image
  Inputs:
    patches [np.ndarray] holding square patch data of shape
      [num_patches, patch_height, patch_width, patch_chan]
    im_shape [list or tuple] containing the output image shape
      [num_im, im_height, im_width, im_chan]
      im_chan must equal patch_chan
      patches must evenly split into im_shape
      can also be [im_height, im_width, im_chan], in which case it is assumed num_im=1
  Outputs:
    images [np.ndarray] of images of shape im_shape
  """
  num_patches, patch_height, patch_width, patch_chan = patches.shape
  if len(im_shape) == 4:
    num_im, im_height, im_width, im_chan = im_shape
  elif len(im_shape) == 3:
    num_im = 1
    im_height, im_width, im_chan = im_shape
    im_shape = [num_im, im_height, im_width, im_chan]
  else:
    assert False, ("input im_shape must have len 3 or 4")
  assert im_height%patch_height == 0, ("Patch height must divide evenly into the image.")
  assert im_width%patch_width == 0, ("Patch width must divide evenly into the image.")
  num_row_patches = np.floor(im_height / patch_height)
  num_col_patches = np.floor(im_width / patch_width)
  num_patches_per_im = int(num_row_patches * num_col_patches)
  tot_num_patches =  int(num_patches_per_im * num_im)
  im_list = [None,]*num_im
  patch_id = 0
  for im_id in range(num_im):
    im_list[im_id] = patches_to_single_image(patches[patch_id:patch_id+num_patches_per_im, ...],
      im_shape[1:])
    patch_id += num_patches_per_im
  images = np.stack(im_list)
  return images

def downsample_data(data, scale_factor, order):
  """Downsample data"""
  return scipy.ndimage.interpolation.zoom(data, scale_factor, order=order, mode="constant")

def rescale_data_to_one(data):
  """
  Rescale input data to be between 0 and 1, per example
  Inputs:
    data: [np.ndarray] unnormalized data
  Outputs:
    data: [np.ndarray] centered data of shape (n, i, j, k) or (n, l)
  """
  data = reshape_data(data, flatten=None)[0]
  data_axis=tuple(range(data.ndim)[1:])
  data_min = np.min(data, axis=data_axis, keepdims=True)
  data_max = np.max(data, axis=data_axis, keepdims=True)
  return (data - data_min) / (data_max - data_min + 1e-6)
  
def normalize_data_with_max(data):
  """
  Normalize data by dividing by abs(max(data))
  Inputs:
    data: [np.ndarray] data to be normalized
  Outputs:
    norm_data: [np.ndarray] normalized data
    data_max: [float] max that was divided out
  """
  if np.max(np.abs(data)) > 0:
    data_max = np.max(np.abs(data))
    norm_data = data / data_max
  else:
    norm_data = data
  return norm_data, data_max

def center_data(data, use_dataset_mean=False):
  """
  Subtract individual example mean from data
  Inputs:
    data: [np.ndarray] unnormalized data of shape:
      (n, i, j, k) - n data points, each of shape (i,j,k), with k channels
      (i, j, k) - single data point of shape (i,j,k)
        Note: output will be reshaped to (1, i, j, k)
      (n, l) - n data points, each of length l
      (l) - single data point of length l
        Note: output will be reshaped to (1, l)
  Outputs:
    data: [np.ndarray] centered data of shape (n, i, j, k) or (n, l)
  """
  if use_dataset_mean or data.ndim == 1:
    data_mean = np.mean(data)
    data -= data_mean
  else:
    data = reshape_data(data, flatten=None)[0] # reshapes to 4D (not flat) or 2D (flat)
    data_axis=tuple(range(data.ndim)[1:])
    data_mean = np.mean(data, axis=data_axis, keepdims=True)
    data -= data_mean
  return data, data_mean

def standardize_data(data):
  """
  Standardize each image data to have zero mean and unit standard-deviation (z-score)
  Inputs:
    data: [np.ndarray] unnormalized data of shape
  Outputs:
    data: [np.ndarray] normalized data
  """
  if data.ndim == 1:
      data -= np.mean(data)
      data /= np.std(data)
  else:
    data = reshape_data(data, flatten=None)[0] # reshapes to 4D (not flat) or 2D (flat)
    data_axis=tuple(range(data.ndim)[1:])
    data_std = np.maximum(np.std(data, axis=data_axis, keepdims=True), 1.0/np.sqrt(data[0,...].size))
    data_mean = np.mean(data, axis=data_axis, keepdims=True)
    data = (data - data_mean) /  data_std
  return data, data_mean, data_std

def normalize_data_with_var(data):
  """
  Divide data by its variance
  Inputs:
    data: [np.ndarray] normalized data of shape:
      (n, i, j) - n data points, each of shape (i,j)
      (n, k) - n data points, each of length k
      (k) - single data point of length k
  Outputs:
    data: [np.ndarray] input data batch
  """
  if data.ndim == 1:
    data_var = np.var(data)
    data /= data_var
  else:
    data_axis=tuple(range(data.ndim)[1:])
    data_var = np.var(data, axis=data_axis, keepdims=True)
    data /= data_var
  return data, data_var

def whiten_data(data, method="FT"):
  """
  Whiten data
  Inputs:
    data: [np.ndarray] with shape [num_examples, height, width, chan]
    method: [str] method to use, can be {FT, PCA, ZCA}
  Outputs:
    whitened_data [np.ndarray]
    data_mean [np.ndarray]
    w_filter [list or np.ndarray] information necessary for unwhitenening
      if method=="FT", then w_filter is np.ndarray representing fourier filter
      if method=="PCA" or "ZCA", then w_filter is a list containing [u, diag(s)] of SVD of covariance matrix
  """
  if method.upper() == "FT":
    flatten=False
    (data, orig_shape, num_examples, num_rows) = reshape_data(data, flatten)[0:4]
    data, data_mean = center_data(data, use_dataset_mean=False)
    data = np.fft.fftshift(np.fft.fft2(data, axes=(1,2,3)), axes=(1,2,3))
    nyq = np.int32(np.floor(num_rows/2))
    freqs = np.linspace(-nyq, nyq-1, num=num_rows)
    fspace = np.meshgrid(freqs, freqs)
    rho = np.sqrt(np.square(fspace[0]) + np.square(fspace[1]))
    lpf = np.exp(-0.5 * np.square(rho / (0.7 * nyq)))
    w_filter = np.multiply(rho, lpf) # filter is in the frequency domain
    data = np.multiply(data, w_filter[None, ..., None])
    data_wht = np.real(np.fft.ifft2(np.fft.ifftshift(data, axes=(1,2,3)), axes=(1,2,3)))
  elif method.upper() == "PCA":
    flatten=True
    (data, orig_shape, num_examples, num_rows) = reshape_data(data, flatten)[0:4]
    data, data_mean = center_data(data, use_dataset_mean=False)
    cov = np.divide(np.dot(data.T, data), num_examples)
    u, s, v = np.linalg.svd(cov) # s are singular values, sqrt(s) are eigenvalues
    isqrtS = np.diag(1 / np.sqrt(s+1e-8))
    w_filter = [u, np.diag(np.sqrt(s+1e-8))] # filter components
    data_wht = np.dot(data, np.dot(u, isqrtS)) 
  elif method.upper() == "ZCA":
    flatten=True
    (data, orig_shape, num_examples, num_rows) = reshape_data(data, flatten)[0:4]
    data, data_mean = center_data(data, use_dataset_mean=False)
    cov = np.divide(np.dot(data.T, data), num_examples)
    u, s, v = np.linalg.svd(cov)
    isqrtS = np.diag(1 / np.sqrt(s+1e-8))
    w_filter = [u, np.diag(np.sqrt(s+1e-8))] # filter components
    data_wht = np.dot(data, np.dot(u, np.dot(isqrtS, u.T)))
  else:
    assert False, ("whitening method must be 'FT', 'ZCA', or 'PCA'")
  if data_wht.shape != orig_shape:
    data_wht = reshape_data(data_wht, not flatten, out_shape=orig_shape[1:])[0]
  return data_wht, data_mean, w_filter

def unwhiten_data(data, data_mean, w_filter, method="FT"):
  """
  Unwhiten data
  Inputs:
    data: [np.ndarray] whitened data with first dim indicating batch
    data_mean: [np.ndarray] data mean (computed before whitening)
    w_filter: [np.ndarray] whitening filter to be inverted
      if method=="FT", then w_filter is np.ndarray representing fourier filter
      if method=="PCA" or "ZCA", then w_filter is a list containing [u, diag(s)] of SVD of covariance matrix
    method: [str] method to use, can be {FT, PCA, ZCA}
  Outputs:
    unwhitened_data
  """
  if method.upper() == "FT":
    flatten=False
    (data, orig_shape, num_examples, num_rows) = reshape_data(data, flatten)[0:4]
    data = np.fft.fftshift(np.fft.fft2(data, axes=(1,2,3)), axes=(1,2,3))
    data = np.multiply(data, (w_filter[None, ..., None]+1e-8)**-1)
    data = np.real(np.fft.ifft2(np.fft.ifftshift(data, axes=(1,2,3)), axes=(1,2,3)))
    data += data_mean
  elif method.upper() == "PCA":
    flatten=True
    (data, orig_shape, num_examples, num_rows) = reshape_data(data, flatten)[0:4]
    u, sqrtS = w_filter
    data = np.dot(data, np.dot(u, sqrtS).T)
  elif method.upper() == "ZCA":
    flatten=True
    (data, orig_shape, num_examples, num_rows) = reshape_data(data, flatten)[0:4]
    u, s = w_filter
    unwhiten_filter = np.dot(np.dot(u, s), u.T)
    data = np.dot(data, unwhiten_filter)
  else:
    assert False, ("whitening method must be 'FT', 'PCA', or 'ZCA'")
  data += data_mean
  if data.shape != orig_shape:
    data = reshape_data(data, not flatten, out_shape=orig_shape[1:])[0]
  return data

def generate_local_contrast_normalizer(radius=12):
  """
  Returns a symmetric Gaussian with specified radius
  Inputs:
    radius: [int] radius of Gaussian function
  Outputs:
    gauss: [np.ndarray] Gaussian filter
  """
  xs = np.linspace(-radius, radius-1, num=2*radius)
  xs, ys = np.meshgrid(xs, xs)
  gauss = np.exp(-0.5*((np.square(xs)+np.square(ys))/radius**2))
  gauss = gauss/np.sum(gauss)
  return gauss

def contrast_normalize(data, gauss_patch_size=12):
  """
  Perform patch-wise local contrast normalization on input data
  Inputs:
    data: [np.ndarray] of shape:
      (n, i, j) - n data points, each of shape (i,j)
      (n, k) - n data points, each of length k
      (k) - single data point of length k
    gauss_patch_size: [int] indicates radius of Gaussian function
  """
  (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
    flatten=False) # Need spatial dim for 2d-Fourier transform
  pooler = generate_local_contrast_normalizer(gauss_patch_size)
  for ex in range(num_examples):
    example = data[ex, ...]
    localIntensityEstimate = scipy.signal.convolve2d(np.square(example), pooler, mode='same')
    normalizedData = np.divide(example, np.sqrt(localIntensityEstimate))
    data[ex, ...] = normalizedData
  if data.shape != orig_shape:
    data = reshape_data(data, flatten=True, out_shape=orig_shape[1:])[0]
  return data

def pca_reduction(data, num_pcs=-1):
  """
  Perform PCA dimensionality reduction on input data
  Inputs:
    data: [np.ndarray] data to be PCA reduced
    num_pcs: [int] number of principal components to keep (-1 for all)
  outputs:
    data_reduc: [np.ndarray] data with reduced dimensionality
  """
  (data, orig_shape, num_examples, num_rows, num_cols, num_channels) = reshape_data(data,
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
  return data_reduc

def compute_power_spectrum(data):
  """
  Compute Fourier power spectrum for input data
  Inputs:
    data: [np.ndarray] of shape:
      (n, i, j) - n data points, each of shape (i,j)
      (n, k) - n data points, each of length k
      (k) - single data point of length k (k must have even sqrt)
  Outputs:
    power_spec: [np.ndarray] Fourier power spectrum
  """
  (data, orig_shape, num_examples, num_rows, num_cols, num_channels) = reshape_data(data,
    flatten=False)
  data = standardize_data(data)[0]
  dataFT = np.fft.fftshift(np.fft.fft2(data, axes=(1, 2)), axes=(1, 2))
  power_spec = np.multiply(dataFT, np.conjugate(dataFT)).real
  return power_spec

def phase_avg_pow_spec(data):
  """
  Compute phase average of power spectrum
  Only works for greyscale imagery
  Inputs:
    data: [np.ndarray] of shape:
      (n, i, j) - n data points, each of shape (i,j)
      (n, k) - n data points, each of length k
      (k) - single data point of length k (k must have even sqrt)
  Outputs:
    phase_avg: [list of np.ndarray] phase averaged power spectrum
      each element in the list corresponds to a data point
  """
  (data, orig_shape, num_examples, num_rows, num_cols, num_channels) = reshape_data(data,
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
