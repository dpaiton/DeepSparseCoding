"""Module for computing contour lines with Euler continuation
"""
import autograd
import autograd.numpy as np

from . import util


def contin(starting_point, f, lims=[0, 1], eps=1e-5, tolerance=1.):
    grad = autograd.grad(f)
    points = [starting_point]

    point = points[-1]
    while (max(point) < max(lims)) & (min(point) > min(lims)):
        contour_vec = util.orthogonalize(grad(point))
        if np.linalg.norm(contour_vec) < 1e-10:
            break
        points.append(point + eps * contour_vec)
        point = points[-1]
        assert np.abs(f(points[-2]) - f(points[-1])) < tolerance

    return np.asarray(points)


def trace_contour(starting_point, f, lims=[0, 1], eps=1e-5, tolerance=1.):
    pos_half_contour = contin(starting_point, f, lims, eps, tolerance)
    neg_half_contour = contin(starting_point, f, lims, -eps, tolerance)
    contour = np.vstack([pos_half_contour[::-1, :], neg_half_contour])

    return contour


def calculate_activation_contours(f, mn=0.05, mx=0.3, N=5,
                                  contour_lims=[-0.5, 1],
                                  contour_eps=1e-2, contour_tolerance=1.):

    """
    Compute contours of f by euler continuation from starting points
    linearly spaced along the x-axis between mn and mx.
    """
    starting_points = [np.asarray([x_val, 0])
                       for x_val in np.linspace(mn, mx, num=N)]

    activation_contours = [trace_contour(starting_point,
                                         f,
                                         lims=contour_lims,
                                         eps=contour_eps,
                                         tolerance=contour_tolerance)
                           for starting_point in starting_points]

    contour_vals = [f(starting_point)
                    for starting_point in starting_points]

    return activation_contours, contour_vals
