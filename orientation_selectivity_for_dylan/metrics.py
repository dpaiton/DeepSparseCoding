"""
This contains implementations for orientation tuning curve metrics
"""
import numpy as np

import spencers_stuff.plotting as sp

def compute_fwhm(centered_ot_curve, corresponding_angles_deg):
  """
  Calculates the full width at half maximum of the tuning curve

  Result is expressed in degrees to make it a little more intuitive. The curve
  is often NOT symmetric about the maximum value so we don't do any fitting and
  we return the FULL width

  Parameters
  ----------
  centered_ot_curve : ndarray
      A 1d array of floats giving the value of the ot curve, at an orientation
      relative to the *preferred orientation* which is given by the angles in
      corresponding_angles_deg. This has the maximum orientation in the
      center of the array which is nicer for visualization.
  corresponding_angles_deg : ndarray
      The orientations relative to preferred orientation that correspond to
      the values in centered_ot_curve

  Returns
  -------
  half_max_left : float
      The position of the intercept to the left of the max
  half_max_right : float
      The position of the intercept to the right of the max
  half_max_value : float
      Mainly for plotting purposes, the actual curve value that corresponds
      to the left and right points
  """
  max_idx = np.argmax(centered_ot_curve)
  min_idx = np.argmin(centered_ot_curve)
  max_val = centered_ot_curve[max_idx]
  min_val = centered_ot_curve[min_idx]
  midpoint = (max_val / 2) + (min_val / 2)
  # find the left hand point
  idx = max_idx
  while centered_ot_curve[idx] > midpoint:
    idx -= 1
    if idx == -1:
      # the width is *at least* 90 degrees
      half_max_left = -90.
      break
  if idx > -1:
    # we'll linearly interpolate between the two straddling points
    # if (x2, y2) is the coordinate of the point below the half-max and
    # (x1, y1) is the point above the half-max, then we can solve for x3, the
    # x-position of the point that corresponds to the half-max on the line
    # that connects (x1, y1) and (x2, y2)
    half_max_left = (((midpoint - centered_ot_curve[idx]) *
                      (corresponding_angles_deg[idx+1] -
                       corresponding_angles_deg[idx]) /
                      (centered_ot_curve[idx+1] - centered_ot_curve[idx])) +
                     corresponding_angles_deg[idx])
  # find the right hand point
  idx = max_idx
  while centered_ot_curve[idx] > midpoint:
    idx += 1
    if idx == len(centered_ot_curve):
      # the width is *at least* 90
      half_max_right = 90.
      break
  if idx < len(centered_ot_curve):
    # we'll linearly interpolate between the two straddling points again
    half_max_right = (((midpoint - centered_ot_curve[idx-1]) *
                       (corresponding_angles_deg[idx] -
                        corresponding_angles_deg[idx-1]) /
                       (centered_ot_curve[idx] - centered_ot_curve[idx-1])) +
                      corresponding_angles_deg[idx-1])

  return half_max_left, half_max_right, midpoint


def compute_circ_var(centered_ot_curve, corresponding_angles_rad):
  """
  Computes the circular variance of a tuning curve and returns vals for plotting

  This is a scale-invariant measure of how 'oriented' a curve is in some
  global sense. It wraps reponses around the unit circle and then sums their
  vectors, resulting in an average vector, the magnitude of which indicates
  the strength of the tuning. Circular variance is an index of 'orientedness'
  that falls in the interval [0.0, 1.0], with 0.0 indicating a delta function
  and 1.0 indicating a completely flat tuning curve.

  Parameters
  ----------
  centered_ot_curve : ndarray
      A 1d array of floats giving the value of the ot curve, at an orientation
      relative to the *preferred orientation* which is given by the angles in
      corresponding_angles_rad. This has the maximum orientation in the
      center of the array which is nicer for visualization.
  corresponding_angles_rad : ndarray
      The orientations relative to preferred orientation that correspond to
      the values in centered_ot_curve

  Returns
  -------
  numerator_sum_components : ndarray
      The complex values the are produced from r * np.exp(j*2*theta). These
      are the elements that get summed up in the numerator
  direction_vector : complex64 or complex128
      This is the vector that points in the direction of *aggregate* tuning.
      its magnitude is upper bounded by 1.0 which is the case when only one
      orientation has a nonzero value. We can plot it to get an idea of how
      tuned a curve is
  circular_variance : float
      This is 1 minus the magnitude of the direction vector. It represents and
      index of 'global selectivity'
  """
  # in the original definition, angles are [0. 2*np.pi] so the factor of 2
  # in the exponential wraps the phase twice around the complex circle,
  # placing responses that correspond to angles pi degrees apart
  # onto the same place. We know there's a redudancy in our responses at pi
  # offsets so our responses get wrapped around the unit circle once.
  numerator_sum_components = (centered_ot_curve *
                              np.exp(1j * 2 * corresponding_angles_rad))
  direction_vector = (np.sum(numerator_sum_components) /
                      np.sum(centered_ot_curve))
  return (numerator_sum_components, direction_vector,
          1.0 - np.abs(direction_vector))


def compute_OSI(centered_ot_curve):
  """
  Compute the Orientation Selectivity Index.

  This is the most coarse but popular measure of selectivity. It really
  doesn't tell you much. It just measures the maximum response relative to
  the minimum response.

  Parameters
  ----------
  centered_ot_curve : ndarray
      A 1d array of floats giving the value of the ot curve, at an orientation
      relative to the *preferred orientation*

  Returns
  -------
  osi : float
      This is (a_max - a_orth) / (a_max + a_orth) where a_max is the maximum
      response across orientations when orientation responses are
      *averages* over phase. a_orth is the orientation which is orthogonal to
      the orientation which produces a_max.
  """
  max_val = np.max(centered_ot_curve)
  # Assume that orthogonal orientation is at either end of the curve modulo 1
  # bin (if we had like an even number of orientation values)
  orth_val = centered_ot_curve[0]
  return (max_val - orth_val) / (max_val + orth_val)


def compute_ot_metrics(ot_dictionary, which_metrics, **kwargs):
  """
  On each of the sets of tuning curves in ot_dictionary compute summary metrics

  Parameters
  ----------
  ot_dictionary : dictionary
      Keys denote the kind of tuning curve it is, mainly how different phases
      were used in calculating the curve or whether we're restricting to
      positive or negative responses only. Examples would include
      'abs_max_over_phase' or 'pos_mean_over_phase', for instance. See
      analysis/base_analysis.spencer_orientation_tuning() for possible
      variants/schemes.
  which_metrics : dictionary(list(str))
      Each key matches the keys in ot_dictionary and the strings in the
      corresponding list give the specific requested metrics.
      Values for these are often lists of floats just to make things easy
  specific_contrast : int, optional
      An index into the contrast dimension of ot_dictionary that gives the
      contrast that we would like to have the metrics computed over
  corresponding_angles_deg : ndarray, optional
      The angles in degrees that are used for all variants in ot_dictionary
  corresponding_angles_rad : ndarray, optional
      The angles in radians that are used for all variants in ot_dictionary

  Returns
  -------
  metrics_dictionary : dictionary(dictionary(list))
      Toplevel keys are the same as in ot_dictionary. Second level keys are
      the requested metrics. Corresponding lists are the values of that metric
      for each basis function
  """
  print("------- Computing orientation tuning metrics -------")
  assert set(ot_dictionary.keys()) == set(which_metrics.keys())
  if 'specific_contrast' in kwargs:
    contrast_idx = kwargs['specific_contrast']
  else:
    contrast_idx = -1
  metrics_dictionary = dict.fromkeys(ot_dictionary)
  for dset_label in ot_dictionary:
    metrics_dictionary[dset_label] = dict.fromkeys(which_metrics[dset_label])
    for metric_label in which_metrics[dset_label]:
      temp = []
      for bf_idx in range(ot_dictionary[dset_label].shape[0]):
        if metric_label == 'full width half maximum':
          assert 'corresponding_angles_deg' in kwargs
          temp.append(compute_fwhm(
            sp.center_curve(ot_dictionary[dset_label][bf_idx, contrast_idx]),
            kwargs['corresponding_angles_deg']))
            # ^compute this on highest provided contrast
        elif metric_label == 'circular variance':
          assert 'corresponding_angles_rad' in kwargs
          temp.append(compute_circ_var(
            sp.center_curve(ot_dictionary[dset_label][bf_idx, contrast_idx]),
            kwargs['corresponding_angles_rad']))
            # ^compute this on highest provided contrast
        elif metric_label == 'orientation selectivity index':
          temp.append(compute_OSI(sp.center_curve(
            ot_dictionary[dset_label][bf_idx, contrast_idx])))
        else:
          raise KeyError('Unrecognized metric: ' + metric_label)

      metrics_dictionary[dset_label][metric_label] = temp

  return metrics_dictionary


