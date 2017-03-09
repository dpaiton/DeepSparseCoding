import numpy as np

"""
Helper function to reshape input data for processing and return data shape
Outputs:
  tuple containing:
  data [np.ndarray] data with new shape
    (num_examples, num_rows, num_cols) if flatten_data==False
    (num_examples, num_elements) if flatten_data==True
  orig_shape [tuple of int32] original shape of the input data
  num_examples [int32] number of data examples
  num_rows [int32] number of data rows (sqrt of num elements)
  num_cols [int32] number of data cols (sqrt of num elements)
Inputs: 
  data [np.ndarray] unnormalized data of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k
  flatten_data [bool] if True, 
"""
def reshape_data(data, flatten_data=False):
  orig_shape = data.shape
  orig_ndim = data.ndim
  if orig_ndim == 1:
    num_examples = 1
    num_elements = data.shape[0]
    assert np.floor(np.sqrt(num_elements)) == np.ceil(np.sqrt(num_elements)), (
      "Data length must have an even square root for spatial whitening.")
    num_rows = np.int32(np.floor(np.sqrt(num_elements)))
    num_cols = num_rows
    if flatten_data:
      data = data[np.newaxis, ...]
    else:
      data = data.reshape((num_rows, num_cols))[np.newaxis, ...]
  elif orig_ndim == 2:
    (num_examples, num_elements) = data.shape
    assert np.floor(np.sqrt(num_elements)) == np.ceil(np.sqrt(num_elements)), (
      "Data length must have an even square root for spatial whitening.")
    num_rows = np.int32(np.floor(np.sqrt(num_elements)))
    num_cols = num_rows
    if not flatten_data:
      data = data.reshape((num_examples, num_rows, num_cols))
  elif orig_ndim == 3:
    (num_examples, num_rows, num_cols) = data.shape
    assert num_rows == num_cols, ("Data points must be square.")
    if flatten_data:
      data = data.reshape((num_examples, num_rows * num_cols))
  else:
    assert False, ("Data must have 1, 2, or 3 dimensions.")
  return (data, orig_shape, num_examples, num_rows, num_cols)


"""
Standardize data to have zero mean and  unit variance
Outputs:
  data [np.ndarray] normalized data
Inputs:
  data [np.ndarray] unnormalized data of shape:
    (n, i, j) - n data points, each of shape (i,j)
    (n, k) - n data points, each of length k
    (k) - single data point of length k
"""
def standardize_data(data):
  data = data[np.newaxis, ...] if data.ndim == 1 else data
  data -= np.mean(data)
  for idx in range(data.shape[0]):
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
  method [str] method to use, can be {FT, PCA}
"""
def whiten_data(data, method="PCA"):
  if method == "FT":
    (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
      flatten_data=False) # Need spatial dim for 2d-Fourier transform
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
    data_wht = reshape_data(data_wht, flatten_data=True)[0]
  elif method == "PCA":
    (data, orig_shape, num_examples, num_rows, num_cols) = reshape_data(data,
      flatten_data=True)
    data -= data.mean(axis=(1))[:,None]
    Cov = np.cov(data.T) # Covariace matrix
    U, S, V = np.linalg.svd(Cov) # SVD decomposition
    isqrtS = np.diag(1 / np.sqrt(S)) # Inverse sqrt of S
    data_wht = np.dot(data, np.dot(np.dot(U, isqrtS), V))
  else:
    assert False, ("whitening method must be 'FT' or 'PCA'")
  return data_wht

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
    flatten_data=False)
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
    flatten_data=False)
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
