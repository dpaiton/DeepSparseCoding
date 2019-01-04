from tests.models.build_test import BuildTest
from tests.models.run_test import RunTest
from tests.utils.contrast_normalize_test import ContrastNormalizeDataTest
from tests.utils.patches_test import PatchesTest
from tests.utils.reshape_data_test import ReshapeDataTest
from tests.utils.standardize_data_test import StandardizeDataTest
from tests.utils.checkpoint_test import CheckpointTest
import tensorflow as tf

import sys

run_all = False
for idx, arg in enumerate(sys.argv):
  #Find this flag, if exists, remove from argv so tensorflow testing args don't
  #mess up
  if arg == "--all":
    run_all = True
    del sys.argv[idx]
    break

if run_all:
  from tests.models.all_test import RunAll

tf.test.main()

