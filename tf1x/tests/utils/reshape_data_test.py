import numpy as np
import tensorflow as tf
import utils.data_processing as dp

"""
Test for the data_processing.reshape_data function.
NOTE: Should be executed from the repository's root directory
function call: reshape_data(data, flatten=None, out_shape=None):
24 possible conditions:
  data: [np.ndarray] data of shape:
      n is num_examples, i is num_rows, j is num_cols, k is num_channels, l is num_examples = i*j*k
      (l) - single data point of shape l, assumes 1 color channel
      (n, l) - n data points, each of shape l (flattened)
      (i, j, k) - single datapoint of of shape (i, j, k)
      (n, i, j, k) - n data points, each of shape (i,j,k)
  flatten: True, False, None
  out_shape: None or (list or tuple)
"""
class ReshapeDataTest(tf.test.TestCase):
  def testBasic(self):
    # mutable parameters
    num_examples_list = [1, 5] # there are conditions where this is assumed 1 and therefore ignored
    num_rows_list = [4]
    num_channels_list = [1, 3]
    # immutable parameters
    num_cols_list = num_rows_list # assumed square objects
    flatten_list = [None, True, False]
    gpu_args = [True, False] if tf.test.is_gpu_available(cuda_only=True) else [False]
    for use_gpu in gpu_args:
      with self.session(use_gpu=use_gpu):
        for num_examples in num_examples_list:
          orig_num_examples = num_examples
          for num_rows, num_cols in zip(num_rows_list, num_cols_list): #assumed to be square
            for num_channels in num_channels_list:
              num_elements = num_rows*num_cols*num_channels
              input_array_list = [
                np.zeros((num_elements)), # assumed num_examples == 1
                np.zeros((num_examples, num_elements)),
                np.zeros((num_rows, num_cols, num_channels)), # assumed num_examples == 1
                np.zeros((num_examples, num_rows, num_cols, num_channels))]
              for input_array in input_array_list:
                input_shape = input_array.shape
                input_ndim = input_array.ndim
                if input_ndim == 1 or input_ndim == 3: # assign num_examples to 1
                  num_examples = 1
                  out_shape_list = [
                    None,
                    (num_elements,),
                    (num_rows, num_cols, num_channels)]
                  if num_channels == 1:
                    out_shape_list.append((num_rows, num_cols))
                else:
                  num_examples = orig_num_examples
                  out_shape_list = [
                    None,
                    (num_examples, num_elements),
                    (num_examples, num_rows, num_cols, num_channels)]
                  if num_channels == 1:
                    out_shape_list.append((num_examples, num_rows, num_cols))
                for out_shape in out_shape_list:
                  for flatten in flatten_list:
                    if out_shape is None and flatten == False and num_channels != 1:
                      # This condition is ill-posed, so the function assumes the image is square
                      # with num_channels == 1. Other conditions will not be tested.
                      continue
                    err_msg = ("\nnum_examples="+str(num_examples)+"\nnum_rows="+str(num_rows)
                      +"\nnum_cols="+str(num_cols)+"\nnum_channels="+str(num_channels)
                      +"\ninput_shape="+str(input_shape)+"\ninput_ndim="+str(input_ndim)
                      +"\nout_shape="+str(out_shape)+"\nflatten="+str(flatten))
                    reshape_outputs = dp.reshape_data(input_array, flatten, out_shape)
                    self.assertEqual(len(reshape_outputs), 6)
                    reshaped_array = reshape_outputs[0]
                    err_msg += "\nreshaped_array.shape="+str(reshaped_array.shape)
                    self.assertEqual(reshape_outputs[1], input_shape, err_msg) # orig_shape
                    #TODO: Add conditional assertions for these outputs - it's more tricky than this
                    #self.assertEqual(reshape_outputs[2], expected_out_shape[0], err_msg) # num_examples
                    #self.assertEqual(reshape_outputs[3], expected_out_shape[1], err_msg) # num_rows
                    #self.assertEqual(reshape_outputs[4], expected_out_shape[2], err_msg) # num_cols
                    #self.assertEqual(reshape_outputs[5], expected_out_shape[3], err_msg) # num_channels
                    if out_shape is None:
                      if flatten is None:
                        if input_ndim == 1 or input_ndim == 3: #ignore num_examples
                          expected_out_shape = tuple([num_examples]+list(input_shape))
                          err_msg += ("\nexpected_out_shape="+str(expected_out_shape))
                          self.assertEqual(reshaped_array.shape, expected_out_shape, err_msg)
                        else:
                          expected_out_shape = input_shape
                          err_msg += ("\nexpected_out_shape="+str(expected_out_shape))
                          self.assertEqual(reshaped_array.shape, expected_out_shape, err_msg)
                      elif flatten == True:
                        expected_out_shape = (num_examples, num_elements)
                        err_msg += ("\nexpected_out_shape="+str(expected_out_shape))
                        self.assertEqual(reshaped_array.shape, expected_out_shape, err_msg)
                      elif flatten == False:
                        expected_out_shape = (num_examples, num_rows, num_cols, num_channels)
                        err_msg += ("\nexpected_out_shape="+str(expected_out_shape))
                        self.assertEqual(reshaped_array.shape, expected_out_shape, err_msg)
                      else:
                        self.assertTrue(False)
                    else:
                      expected_out_shape = out_shape
                      err_msg += ("\nexpected_out_shape="+str(expected_out_shape))
                      self.assertEqual(reshaped_array.shape, expected_out_shape, err_msg)

if __name__ == "__main__":
  tf.test.main()
