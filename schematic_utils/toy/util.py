from itertools import product

import autograd
import autograd.numpy as np


def orthogonalize(twovector):
    return np.asarray([-twovector[1], twovector[0]])


def follow_grad(starting_point, f, lims=[0, 1], eps=1e-2):
    grad = autograd.grad(f)
    points = [starting_point]

    point = points[-1]
    while (max(point) < max(lims)) & (min(point) > min(lims)):
        update = grad(point)

        if np.linalg.norm(update) < 1e-10:
            break
        points.append(point + eps * update)
        point = points[-1]

    return np.asarray(points)


def normalize_dict(dictionary):
    normalized_dict_elems = [
        normalize(dict_elem) for dict_elem in dictionary.T]
    return np.asarray(normalized_dict_elems).T


def normalize(dictionary_element):
    return dictionary_element / np.linalg.norm(dictionary_element)


def compute_grads(f, xlims, ylims, xN, yN):
    xs = np.linspace(*xlims, num=xN)
    ys = np.linspace(*ylims, num=yN)

    grads = [autograd.grad(f)(point) for point in product(xs, ys)]

    return grads, list(product(xs, ys))
