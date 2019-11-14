from analysis.base_analyzer import Analyzer

class MlpAnalyzer(Analyzer):
  def __init__(self):
    super(MlpAnalyzer, self).__init__()
    self.var_names = ["mlp/layer0/fc_w_0:0"]

  def run_analysis(self, images, labels, save_info=""):
    super(MlpAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.analysis_params.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["mlp/layer0/fc_w_0:0"], save_info)
    if self.analysis_params.do_class_adversaries:
      #TODO generate target labels based on options here
      #For now, defined in analysis_params
      self.class_adversary_analysis(images,
        labels, batch_size=self.analysis_params.eval_batch_size,
        input_id=self.analysis_params.adversarial_input_id,
        target_method = self.analysis_params.adversarial_target_method,
        target_labels = self.analysis_params.adversarial_target_labels,
        save_info=save_info)
    elif self.analysis_params.do_neuron_visualization:
      print("Neuron Visualization Analysis")
      self.neuron_visualization_analysis(save_info=save_info)
