import sys
import os

import tensorflow as tf

#Default test runs if no args specified
run_comb = False
run_build = True
run_run = True
run_utils = True
run_analysis = True
run_data = False

del_idx = []
for idx, arg in enumerate(sys.argv):
  #Find this flag, if exists, remove from argv so tensorflow testing args don't mess up
  if arg == "--all":
    run_comb = True
    run_build = True
    run_run = True
    run_utils = True
    run_analysis = True
    run_data = True
    del_idx.append(idx)
  if arg == "--comb":
    run_comb = True
    run_build = False
    run_run = False
    run_utils = False
    run_analysis = False
    run_data = False
    del_idx.append(idx)
  if arg == "--build":
    run_comb = False
    run_build = True
    run_run = False
    run_utils = False
    run_analysis = False
    run_data = False
    del_idx.append(idx)
  if arg == "--run":
    run_comb = False
    run_build = False
    run_run = True
    run_utils = False
    run_analysis = False
    run_data = False
    del_idx.append(idx)
  if arg == "--utils":
    run_comb = False
    run_build = False
    run_run = False
    run_utils = True
    run_analysis = False
    run_data = False
    del_idx.append(idx)
  if arg == "--analysis":
    run_comb = False
    run_build = False
    run_run = False
    run_utils = False
    run_analysis = True
    run_data = False
    del_idx.append(idx)
  if arg == "--data":
    run_comb = False
    run_build = False
    run_run = False
    run_utils = False
    run_analysis = False
    run_data = True
    del_idx.append(idx)


#Remove all del_idxs
for idx in del_idx:
  del sys.argv[idx]

#build_test and run_test all have multiple dymaically created classes, so import *
if run_comb:
  #RunAll_[model_type]_[data_type]
  from tests.models.comb_test import *

if run_build:
  #BuildTest_[model_type]
  from tests.models.build_test import *

if run_run:
  #RunTest_[model_type]
  from tests.models.run_test import *

if run_utils:
  from tests.utils.contrast_normalize_test import ContrastNormalizeDataTest
  from tests.utils.patches_test import PatchesTest
  from tests.utils.reshape_data_test import ReshapeDataTest
  from tests.utils.standardize_data_test import StandardizeDataTest
  from tests.utils.checkpoint_test import CheckpointTest

if run_analysis:
  from tests.analysis.atas_test import ActivityTriggeredAverageTest

if run_data:
  from tests.data.data_selector_test import DataSelectorTest

tf.test.main()
