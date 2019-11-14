import numpy as np

try:
    import vapeplot
    contour_cmap = vapeplot.cmap("macplus")
except ImportError:
    print("No vapeplot, falling back to Greys cmap.")
    contour_cmap = "Greys"


mnist_dims = (28, 28)
mnist_dim = mnist_dims[0] * mnist_dims[1]


def orthogonalize(vector, to_vector):
    """ Unit vectors only"""
    return vector - np.dot(vector, to_vector) * to_vector
