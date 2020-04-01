import numpy as np
import tensorflow as tf

from DeepSparseCoding.tf1x.analysis.base_analyzer import Analyzer

"""
Test for activity triggered analysis

NOTE: Should be executed from the repository's root directory
"""
class ActivityTriggeredAverageTest(tf.test.TestCase):
  def testBasic(self):
    rand_state = np.random.RandomState(1234)
    rand_mean = 2.0
    rand_var = 10
    num_images = 50
    num_pixels = 12
    num_neurons = 24 
    base_analyzer = Analyzer()
    model_weights = rand_state.normal(loc=0.0, scale=1.0, size=(num_pixels, num_neurons))
    images = rand_state.normal(loc=rand_mean, scale=rand_var, size=[num_images, num_pixels])

    # Batch size is greater than num images (shouldn't use batches)
    batch_size = 100
    atas_1 = base_analyzer.compute_atas(images, np.dot(images, model_weights), batch_size)

    # Batch size is less than num images, but divides evenly
    batch_size = 10
    atas_2 = base_analyzer.compute_atas(images, np.dot(images, model_weights), batch_size)

    # Batch size is less than num_images, but does not divide evenly
    batch_size = 13
    atas_3 = base_analyzer.compute_atas(images, np.dot(images, model_weights), batch_size)

    self.assertAllClose(atas_1, atas_2, rtol=1e-06, atol=1e-06)
    self.assertAllClose(atas_1, atas_3, rtol=1e-06, atol=1e-06)

if __name__ == "__main__":
  tf.test.main()
