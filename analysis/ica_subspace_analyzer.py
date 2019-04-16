import numpy as np 
import tensorflow as tf
from analysis.base_analyzer import Analyzer
import utils.data_processing as dp 

class IcaAnalyzer(Analyzer):
    def __init__(self):
        Analyzer.__init__(self)
        self.var_names = [
                "weights/w_synth:0"
                ]


