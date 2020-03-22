from DeepSparseCoding.analysis.base_analyzer import Analyzer
import DeepSparseCoding.utils.data_processing as dp

class RicaAnalyzer(Analyzer):
  def __init__(self):
    Analyzer.__init__(self)
    self.var_names = [
      "weights/w:0",
      "inference/activity:0",
      "output/reconstruction:0"]

  def run_analysis(self, images, labels=None, save_info=""):
    super(RicaAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.analysis_params.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/w:0"], save_info)
    if self.analysis_params.do_atas:
      self.atas, self.atcs = self.ata_analysis(images, self.evals["inference/activity:0"],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.analysis_params.do_orientation_analysis:
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)
