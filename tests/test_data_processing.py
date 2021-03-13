import os
import sys
import unittest
from os.path import dirname as up

ROOT_DIR = up(up(up(os.path.realpath(__file__))))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import torch
import numpy as np

import DeepSparseCoding.utils.data_processing as dp


class TestUtils(unittest.TestCase):
    def test_reshape_data(self):
        """
        Test for the data_processing.reshape_data function.
        NOTE: Should be executed from the repository's root directory
        function call: reshape_data(data, flatten=None, out_shape=None):
        24 possible conditions:
          data: [np.ndarray] data of shape:
              n is num_examples, i is num_channels, j is num_rows, k is num_cols
              (i) - single data point of shape i
              (n, i) - n data points, each of shape i (flattened)
              (i, j, k) - single datapoint of of shape (i, j, k)
              (n, i, j, k) - n data points, each of shape (i,j,k)
          flatten: True, False, None
          out_shape: None or (list or tuple)
        """
        # mutable parameters
        num_examples_list = [1, 5] # there are conditions where this is assumed 1 and therefore ignored
        num_rows_list = [4]
        num_channels_list = [1, 3]
        # immutable parameters
        num_cols_list = num_rows_list # assumed square objects
        flatten_list = [None, True, False]
        for num_examples in num_examples_list:
            orig_num_examples = num_examples
            for num_rows, num_cols in zip(num_rows_list, num_cols_list): #assumed to be square
                for num_channels in num_channels_list:
                    num_elements = num_rows*num_cols*num_channels
                    input_array_list = [
                        np.zeros((num_elements)), # assumed num_examples == 1
                        np.zeros((num_examples, num_elements)),
                        np.zeros((num_channels, num_rows, num_cols)), # assumed num_examples == 1
                        np.zeros((num_examples, num_channels, num_rows, num_cols))]
                    for input_array in input_array_list:
                        input_shape = input_array.shape
                        input_ndim = input_array.ndim
                        if(input_ndim == 1 or input_ndim == 3): # assign num_examples to 1
                            num_examples = 1
                            out_shape_list = [
                                None,
                                (num_elements,),
                                (num_channels, num_rows, num_cols)]
                            if(num_channels == 1):
                                out_shape_list.append((num_rows, num_cols))
                        else:
                            num_examples = orig_num_examples
                            out_shape_list = [
                                None,
                                (num_examples, num_elements),
                                (num_examples, num_channels, num_rows, num_cols)]
                            if(num_channels == 1):
                                out_shape_list.append((num_examples, num_rows, num_cols))
                        for out_shape in out_shape_list:
                            for flatten in flatten_list:
                                if(out_shape is None and flatten == False and num_channels != 1):
                                    # This condition is ill-posed, so the function assumes the image is square
                                    # with num_channels == 1. Other conditions will not be tested.
                                    continue
                                err_msg = (f'\nnum_examples={num_examples}'+f'\nnum_rows={num_rows}'
                                    +f'\nnum_cols={num_cols}'+f'\nnum_channels={num_channels}'
                                    +f'\ninput_shape={input_shape}'+f'\ninput_ndim={input_ndim}'
                                    +f'\nout_shape={out_shape}'+f'\nflatten={flatten}')
                                reshape_outputs = dp.reshape_data(
                                    torch.tensor(input_array),
                                    flatten,
                                    out_shape)
                                self.assertEqual(len(reshape_outputs), 6)
                                reshaped_array = reshape_outputs[0].numpy()
                                err_msg += f'\nreshaped_array.shape={reshaped_array.shape}'
                                self.assertEqual(reshape_outputs[1], input_shape, err_msg) # orig_shape
                                (resh_num_examples, resh_num_channels, resh_num_rows, resh_num_cols) = reshape_outputs[2:]
                                err_msg += (f'\nfunction_shape_outputs={reshape_outputs[2:]}')
                                if(out_shape is None):
                                    if(flatten is None):
                                        if(input_ndim == 1 or input_ndim == 3): #ignore num_examples
                                            expected_out_shape = tuple([num_examples]+list(input_shape))
                                            self.assertEqual(
                                                resh_num_examples,
                                                1,
                                                err_msg)
                                        else:
                                            expected_out_shape = input_shape
                                        err_msg += f'\nexpected_out_shape={expected_out_shape}'
                                        self.assertEqual(reshaped_array.shape, expected_out_shape, err_msg)
                                    elif(flatten == True):
                                        expected_out_shape = (num_examples, num_elements)
                                        err_msg += f'\nexpected_out_shape={expected_out_shape}'
                                        self.assertEqual(
                                            reshaped_array.shape,
                                            expected_out_shape,
                                            err_msg)
                                        self.assertEqual(
                                            resh_num_channels * resh_num_rows * resh_num_cols,
                                            expected_out_shape[1],
                                            err_msg)
                                    elif(flatten == False):
                                        expected_out_shape = (num_examples, num_channels, num_rows, num_cols)
                                        err_msg += f'\nexpected_out_shape={expected_out_shape}'
                                        self.assertEqual(
                                            reshaped_array.shape,
                                            expected_out_shape,
                                            err_msg)
                                        self.assertEqual(
                                            resh_num_channels,
                                            expected_out_shape[1],
                                            err_msg)
                                        self.assertEqual(
                                            resh_num_rows,
                                            expected_out_shape[2],
                                            err_msg)
                                        self.assertEqual(
                                            resh_num_cols,
                                            expected_out_shape[3],
                                            err_msg)
                                    else:
                                        self.assertTrue(False)
                                else:
                                    expected_out_shape = out_shape
                                    err_msg += (f'\nexpected_out_shape={expected_out_shape}')
                                    self.assertEqual(reshaped_array.shape, expected_out_shape, err_msg)
                                    self.assertEqual(resh_num_examples, None, err_msg)

    def test_standardize(self):
        num_tolerance_decimals = 5
        unflat_shape = [8, 4, 4, 3]
        flat_shape = [8, 4*4*3]
        shape_options = [unflat_shape, flat_shape]
        eps_options = [1e-8, None]
        samplewise_options = [True, False]
        for shape in shape_options:
            for eps_val in eps_options:
                for samplewise in samplewise_options:
                    err_msg = (f'\ninput_shape={shape}\neps={eps_val}\nsamplewise={samplewise}')
                    random_tensor = torch.rand(shape)
                    func_output = dp.standardize(random_tensor, eps=eps_val, samplewise=samplewise)
                    norm_tensor = func_output[0].numpy()
                    orig_mean = func_output[1]
                    orig_std = func_output[2]
                    if samplewise:
                        for idx in range(shape[0]):
                            self.assertAlmostEqual(
                                np.mean(norm_tensor[idx, ...]),
                                0.0,
                                places=num_tolerance_decimals,
                                msg=err_msg)
                            self.assertAlmostEqual(
                                np.std(norm_tensor[idx, ...]),
                                1.0,
                                places=num_tolerance_decimals,
                                msg=err_msg)
                    else:
                        self.assertAlmostEqual(
                            np.mean(norm_tensor),
                            0.0,
                            places=num_tolerance_decimals,
                            msg=err_msg)
                        self.assertAlmostEqual(
                            np.std(norm_tensor),
                            1.0,
                            places=num_tolerance_decimals,
                            msg=err_msg)

    def test_rescale_data_to_one(self):
        num_tolerance_decimals = 7
        unflat_shape = [8, 4, 4, 3]
        flat_shape = [8, 4*4*3]
        shape_options = [unflat_shape, flat_shape]
        eps_options = [1e-6, None]
        samplewise_options = [True, False]
        for shape in shape_options:
            for eps_val in eps_options:
                for samplewise in samplewise_options:
                    err_msg = (f'\ninput_shape={shape}\neps={eps_val}\nsamplewise={samplewise}')
                    random_tensor = torch.rand(shape)
                    func_output = dp.rescale_data_to_one(random_tensor, eps=eps_val, samplewise=samplewise)
                    norm_tensor = func_output[0].numpy()
                    orig_min = func_output[1]
                    orig_max = func_output[2]
                    err_msg += (f'\noriginal_min={orig_min}\noriginal_max={orig_max}')
                    if samplewise:
                        for idx in range(shape[0]):
                            self.assertAlmostEqual(
                                np.min(norm_tensor[idx, ...]),
                                0.0,
                                places=num_tolerance_decimals,
                                msg=err_msg)
                            self.assertAlmostEqual(
                                np.max(norm_tensor[idx, ...]),
                                1.0,
                                places=num_tolerance_decimals,
                                msg=err_msg)
                    else:
                        self.assertAlmostEqual(
                            np.min(norm_tensor),
                            0.0,
                            places=num_tolerance_decimals,
                            msg=err_msg)
                        self.assertAlmostEqual(
                            np.max(norm_tensor),
                            1.0,
                            places=num_tolerance_decimals,
                            msg=err_msg)

    def test_label_conversion(self):
        num_labels = 15
        num_classes = 10
        dense_labels = np.random.randint(0, num_classes, num_labels)
        one_hot_labels = np.zeros((num_labels, num_classes))
        for label_index in range(num_labels):
            one_hot_labels[label_index, dense_labels[label_index]] = 1
        func_dense = dp.one_hot_to_dense(torch.tensor(one_hot_labels)).numpy()
        np.testing.assert_equal(func_dense, dense_labels)
        func_one_hot = dp.dense_to_one_hot(torch.tensor(dense_labels), num_classes).numpy()
        np.testing.assert_equal(func_one_hot, one_hot_labels)

    def test_atleastkd(self):
        x = np.random.standard_normal([2, 3, 4])
        ks = [0, 3, 8, 10]
        for k in ks:
            new_x = dp.atleast_kd(torch.tensor(x), k).numpy()
            test_nd = np.maximum(k, x.ndim)
            np.testing.assert_equal(new_x.ndim, test_nd)

    def test_l2_weight_norm(self):
        w_fc = np.random.standard_normal([38, 24])
        w_conv = np.random.standard_normal([38, 24, 8, 8])
        for w in [w_fc, w_conv]:
            w_norm = dp.get_weights_l2_norm(torch.tensor(w), eps=1e-12).numpy()
            normed_w = dp.l2_normalize_weights(torch.tensor(w), eps=1e-12).numpy()
            normed_w_norm = dp.get_weights_l2_norm(torch.tensor(normed_w), eps=1e-12).numpy()
            np.testing.assert_allclose(normed_w_norm, 1.0, rtol=1e-10)
            np.testing.assert_allclose(w / w_norm, normed_w, rtol=1e-10)

    def test_patches(self):
      err = 1e-6
      rand_mean = 0; rand_var = 1
      num_im = 10; im_edge = 512; im_chan = 1; patch_edge = 16
      num_patches = np.int(num_im * (im_edge / patch_edge)**2)
      rand_seed = 1234
      rand_state = np.random.RandomState(rand_seed)
      data = np.stack([rand_state.normal(rand_mean, rand_var, size=[im_chan, im_edge, im_edge])
          for _ in range(num_im)])
      data_shape = list(data.shape)
      patch_shape = [im_chan, patch_edge, patch_edge]
      datapoint = torch.tensor(data[0, ...])
      datapoint_patches = dp.single_image_to_patches(datapoint, patch_shape)
      datapoint_recon = dp.patches_to_single_image(datapoint_patches, data_shape[1:])
      np.testing.assert_allclose(datapoint.numpy(), datapoint_recon.numpy(), rtol=err)
      patches = dp.images_to_patches(torch.tensor(data), patch_shape)
      data_recon = dp.patches_to_images(patches, data_shape[1:])
      np.testing.assert_allclose(data, data_recon.numpy(), rtol=err)
