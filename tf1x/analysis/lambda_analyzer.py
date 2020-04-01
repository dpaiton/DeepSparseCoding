from DeepSparseCoding.analysis.base_analyzer import Analyzer

class LambdaAnalyzer(Analyzer):
  def __init__(self):
    super(LambdaAnalyzer, self).__init__()
    self.var_names = ["mlp/layer0/conv_w_0:0"]

  def run_analysis(self, images, labels, save_info=""):
    super(LambdaAnalyzer, self).run_analysis(images, labels, save_info=save_info)
