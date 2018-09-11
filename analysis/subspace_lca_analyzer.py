import os
import numpy as np
import tensorflow as tf
from analysis.lca_analyzer import LCA_Analyzer
import utils.data_processing as dp

class SUBSPACE_LCA_Analyzer(LCA_Analyzer):
  def __init__(self, params):
    super(SUBSPACE_LCA_Analyzer, self).__init__(params)
