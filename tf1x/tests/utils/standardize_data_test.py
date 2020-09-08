import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.utils.data_processing as dp

"""
Test for the data_processing.standardize_data function.
NOTE: Should be executed from the repository's root directory
"""
class StandardizeDataTest(tf.test.TestCase):
  def testBasic(self):
    err = 1e-4; rand_mean=5; rand_std=3;
    num_examples=10; num_rows=256; num_cols=256; num_channels=1
    rand_state = np.random.RandomState(1234)
    gpu_args = [True, False] if tf.test.is_gpu_available(cuda_only=True) else [False]
    for use_gpu in gpu_args:
      with self.session(use_gpu=use_gpu):
        data = np.stack([rand_state.normal(rand_mean, rand_std,
          size=[num_rows, num_cols, num_channels]) for _ in range(num_examples)], axis=0)
        stand_data, data_mean, data_std = dp.standardize_data(data)
        for idx in range(stand_data.shape[0]):
          self.assertNear(np.mean(stand_data[idx, ...]), 0.0, err)
          self.assertNear(np.std(stand_data[idx, ...]), 1.0, err)
          hand_stand_data = (data[idx, ...] - data_mean[idx]) /  data_std[idx]
          unstandardized_data = (stand_data[idx, ...] * data_std[idx, ...]) + data_mean[idx, ...]
          self.assertAllClose(stand_data[idx, ...], hand_stand_data, rtol=err, atol=err)
          self.assertAllClose(unstandardized_data, data[idx, ...], rtol=err, atol=err)

if __name__ == "__main__":
  tf.test.main()
