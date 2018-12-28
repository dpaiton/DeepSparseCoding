#!/bin/bash
#echo "----TEST 1 of 5"
#CUDA_VISIBLE_DEVICES=0 python3 tests/utils/reshape_data_test.py
#echo "----TEST 2 of 5"
#CUDA_VISIBLE_DEVICES=0 python3 tests/utils/standardize_data_test.py
#echo "----TEST 3 of 5"
#CUDA_VISIBLE_DEVICES=0 python3 tests/utils/patches_test.py
#echo "----TEST 4 of 5"
#CUDA_VISIBLE_DEVICES=0 python3 tests/utils/contrast_normalize_test.py
echo "----TEST 5 of 5"
CUDA_VISIBLE_DEVICES=0 python3 tests/models/models_test.py
#TODO: do this if flag is set
#CUDA_VISIBLE_DEVICES=0 python3 tests/training/training_test.py
