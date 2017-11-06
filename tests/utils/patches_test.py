import numpy as np
import tensorflow as tf
import utils.data_processing as dp

"""
Test for the extract_patches and patches_to_image functions.
NOTE: Should be executed from the repository's root directory
"""
class PatchesTest(tf.test.TestCase):
  def testBasic(self):
    err = 1e-15;
    rand_mean = 0; rand_var = 1
    num_im=10; im_edge=512; patch_edge=16
    num_patches = np.int(num_im * (im_edge/patch_edge)**2)
    rand_state = np.random.RandomState(1234)
    gpu_args = [True, False] if tf.test.is_gpu_available(cuda_only=True) else [False]
    for use_gpu in gpu_args:
      with self.test_session(use_gpu=use_gpu):
        data = rand_state.normal(rand_mean, rand_var, size=[num_im, im_edge, im_edge])
        datapoint = data[0, ...]
        datapoint_patches = dp.extract_patches_from_single_image(datapoint, patch_edge)
        datapoint_recon = dp.patches_to_image(datapoint_patches, 1, im_edge)
        self.assertNear(np.mean(np.abs(datapoint-datapoint_recon)), 0.0, err)
        patches = dp.extract_patches(data, out_shape=(num_patches, patch_edge**2),
          overlapping=False, randomize=False, var_thresh=0, rand_state=rand_state)
        data_recon = dp.patches_to_image(patches, num_im, im_edge)
        self.assertNear(np.mean(np.abs(data-data_recon)), 0.0, err)
        patches = dp.extract_patches(data, out_shape=(num_patches, patch_edge**2),
          overlapping=False, randomize=True, var_thresh=0, rand_state=rand_state)
        self.assertNear(np.mean(patches), np.mean(data), err)
        patches = dp.extract_patches(data, out_shape=(2*num_patches, patch_edge**2),
          overlapping=True, randomize=True, var_thresh=0, rand_state=rand_state)
        self.assertNear(np.mean(patches), np.mean(data), 1e-2)

if __name__ == "__main__":
  tf.test.main()
