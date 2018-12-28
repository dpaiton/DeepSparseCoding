#!/bin/bash
if [ "$1" = "--train" ]; then
  NUM_TEST="6"
else
  NUM_TEST="5"
fi

echo "----TEST 1 of $NUM_TEST"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/reshape_data_test.py
echo "----TEST 2 of $NUM_TEST"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/standardize_data_test.py
echo "----TEST 3 of $NUM_TEST"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/patches_test.py
echo "----TEST 4 of $NUM_TEST"
CUDA_VISIBLE_DEVICES=0 python3 tests/utils/contrast_normalize_test.py
echo "----TEST 5 of $NUM_TEST"
CUDA_VISIBLE_DEVICES=0 python3 tests/models/models_test.py
if [ "$1" == "--train" ]; then
  echo "----TEST 6 of $NUM_TEST"
  CUDA_VISIBLE_DEVICES=0 python3 tests/training/training_test.py
fi
