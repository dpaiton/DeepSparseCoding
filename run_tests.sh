#!/bin/bash
echo "----TEST 1 of 4"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/reshape_data_test.py
echo "----TEST 2 of 4"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/standardize_data_test.py
echo "----TEST 3 of 4"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/patches_test.py
echo "----TEST 4 of 4"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/contrast_normalize_test.py
