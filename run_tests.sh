#!/bin/bash
echo "----TEST 1 of 2"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/standardize_data_test.py
echo "----TEST 2 of 2"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/patches_test.py
