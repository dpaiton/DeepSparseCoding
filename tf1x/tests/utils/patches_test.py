import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.utils.data_processing as dp

"""
Test for the extract_patches and patches_to_image functions.
NOTE: Should be executed from the repository's root directory
"""
class PatchesTest(tf.test.TestCase):
  def testBasic(self):
    err = 1e-15;
    rand_mean = 0; rand_var = 1
    num_im=10; im_edge=512; im_chan=1; patch_edge=16
    num_patches = np.int(num_im * (im_edge/patch_edge)**2)
    rand_state = np.random.RandomState(1234)
    gpu_args = [True, False] if tf.test.is_gpu_available(cuda_only=True) else [False]
    for use_gpu in gpu_args:
      with self.session(use_gpu=use_gpu):
        data = np.stack([rand_state.normal(rand_mean, rand_var, size=[im_edge, im_edge, im_chan])
          for _ in range(num_im)])
        data_shape = list(data.shape)
        patch_shape = [num_patches, patch_edge, patch_edge, im_chan]

        datapoint = data[0, ...]
        datapoint_patches = dp.extract_patches_from_single_image(datapoint, patch_shape[1:])
        datapoint_recon = dp.patches_to_image(datapoint_patches, im_shape=[1]+data_shape[1:])
        self.assertNear(np.mean(np.abs(datapoint-datapoint_recon)), 0.0, err)

        patches = dp.extract_patches(data, out_shape=patch_shape,
          overlapping=False, randomize=False, var_thresh=0, rand_state=rand_state)
        data_recon = dp.patches_to_image(patches, data_shape)
        self.assertNear(np.mean(np.abs(data-data_recon)), 0.0, err)

        patches = dp.extract_patches(data, out_shape=patch_shape,
          overlapping=False, randomize=True, var_thresh=0, rand_state=rand_state)
        self.assertNear(np.mean(patches), np.mean(data), err)

        patches = dp.extract_patches(data, out_shape=[2*num_patches]+patch_shape[1:],
          overlapping=True, randomize=True, var_thresh=0, rand_state=rand_state)
        self.assertNear(np.mean(patches), np.mean(data), 1e-2)

if __name__ == "__main__":
  tf.test.main()
