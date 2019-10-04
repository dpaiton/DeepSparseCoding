from analysis.base_analyzer import Analyzer
import numpy as np

class MlpAnalyzer(Analyzer):
  def __init__(self, is_lca=False):
    self.is_lca = is_lca
    super(MlpAnalyzer, self).__init__()

  def run_analysis(self, images, labels, save_info=""):
    super(MlpAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_class_adversaries:
      #TODO generate target labels based on options here
      #For now, defined in analysis_params
      self.class_adversary_analysis(images,
        labels, batch_size=self.analysis_params.eval_batch_size,
        input_id=self.analysis_params.adversarial_input_id,
        target_method = self.analysis_params.adversarial_target_method,
        target_labels = self.analysis_params.adversarial_target_labels,
        save_info=save_info)
    if self.analysis_params.do_evals:
      out_evals = ["input_node:0", "label_est:0"]
      if(self.is_lca):
        out_evals.append("reconstruction:0")
        out_evals.append("activations:0")

      #Run model
      out_dict = self.evaluate_model_batch(self.analysis_params.eval_batch_size,
          images, out_evals)
      out_dict["labels"] = labels
      out_fn = self.analysis_out_dir+"savefiles/evals_"+save_info+".npz"
      np.savez(out_fn, data={"evals":out_dict})
