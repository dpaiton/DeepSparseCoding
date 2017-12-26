import numpy as np
import tensorflow as tf
import utils.data_processing as dp

"""
Test for the standardize_data function.
NOTE: Should be executed from the repository's root directory
"""
class StandardizeDataTest(tf.test.TestCase):
  def testBasic(self):
    # TODO: err should scale with variance
    err = 1e-15; rand_mean=5.0; rand_var=1000.0;
    num_samples=10; num_rows=16; num_cols=16
    rand_state = np.random.RandomState(1234)
    gpu_args = [True, False] if tf.test.is_gpu_available(cuda_only=True) else [False]
    for use_gpu in gpu_args:
      with self.test_session(use_gpu=use_gpu):
        data = rand_state.normal(rand_mean, rand_var, size=[num_samples, num_rows, num_cols])
        stand_data, data_mean = dp.standardize_data(data)
        for idx in range(stand_data.shape[0]):
          self.assertNear(np.mean(stand_data[idx, ...]), 0.0, err)
          self.assertNear(np.std(data[idx, ...]), 1.0, err)

if __name__ == "__main__":
  tf.test.main()
