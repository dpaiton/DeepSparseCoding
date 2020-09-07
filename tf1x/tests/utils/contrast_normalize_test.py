import os
import sys

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)

import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.utils.data_processing as dp

"""
Test for the contrast_normalize function.
NOTE: Should be executed from the repository's root directory
TODO: Add assertions? How would I test if a data sample is conrast normalized?
"""
class ContrastNormalizeDataTest(tf.test.TestCase):
  def testBasic(self):
    rand_mean=5.0; rand_var=1000.0;
    num_samples=10; num_rows=16; num_cols=16
    rand_state = np.random.RandomState(1234)
    gpu_args = [True, False] if tf.test.is_gpu_available(cuda_only=True) else [False]
    for use_gpu in gpu_args:
      with self.session(use_gpu=use_gpu):
        data = rand_state.normal(rand_mean, rand_var, size=[num_rows, num_cols, 1])
        data_cn = dp.contrast_normalize(data)

        data = np.stack([rand_state.normal(rand_mean, rand_var, size=[num_rows, num_cols, 1])
          for _ in range(num_samples)], axis=0)
        data_cn = dp.contrast_normalize(data)

        data = np.stack([rand_state.normal(rand_mean, rand_var, size=[num_rows, num_cols, 3])
          for _ in range(num_samples)], axis=0)
        data_cn = dp.contrast_normalize(data)

if __name__ == "__main__":
  tf.test.main()
