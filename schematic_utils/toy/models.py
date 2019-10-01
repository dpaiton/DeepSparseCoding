import autograd.numpy as np

default_lam = 5e-2


class LCA(object):

    def __init__(self, dictionary, activation=None,
                 delta_t=1e-1):

        self.dictionary = dictionary
        self.num_inputs, self.num_neurons = dictionary.shape

        if activation is None:
            self.activation = lambda x: sparse_abs(x, default_lam)
        else:
            self.activation = activation

        self.delta_t = delta_t

    def inference_step(self, x, u=None, delta_t=None,
                       drive=None, inhibitory_weights=None):
        if delta_t is None:
            delta_t = self.delta_t
        if drive is None:
            drive = np.dot(self.dictionary.T, x)
        if u is None:
            u = np.zeros_like(drive)

        a = self.activation(u)
        if inhibitory_weights is None:
            inhibitory_weights = self.inhibitory_weights

        inhibition = -np.dot(inhibitory_weights, a)

        return u + delta_t * (drive + inhibition - u)

    def forward_pass(self, x, num_iters=50):
        u = None
        drive = np.dot(self.dictionary.T, x)
        inhibitory_weights = np.copy(self.inhibitory_weights)
        for _ in range(num_iters):
            u = self.inference_step(x, u, drive=drive,
                                    inhibitory_weights=inhibitory_weights)

        return self.activation(u)

    @property
    def gramian(self):
        return np.dot(self.dictionary.T, self.dictionary)

    @property
    def inhibitory_weights(self):
        return self.gramian - np.eye(self.num_neurons)

    def reconstruct(self, a):
        return np.dot(self.dictionary, a)


class MLP(object):

    def __init__(self, weight_matrices, activation="relu"):

        if not isinstance(weight_matrices, list):
            weight_matrices = [weight_matrices]

        self.weight_matrices = weight_matrices

        if activation == "relu":
            self.activation = relu
        else:
            self.activation = activation

    def forward_pass(self, x):
        activations = x
        for weight_matrix in self.weight_matrices:
            activations = self.activation(np.dot(weight_matrix,
                                                 activations))
        return activations


def relu(x):
    return np.where(x > 0., x, 0)


def softplus(x, lam=5.):
    return 1 / lam * np.log(1 + np.exp(lam * x))


def sigmoid(x):
    return np.where(x >= 0, _positive_sigm(x), _negative_sigm(x))


def swish(x):
    return np.multiply(x, sigmoid(x))


def _negative_sigm(x):
    expon = np.exp(-x)
    return 1 / (1 + expon)


def _positive_sigm(x):
    expon = np.exp(x)
    return expon / (1 + expon)


def sparse_abs(x, lam):
    abs_x = np.abs(x)
    return np.where(abs_x < lam, 0, abs_x-lam)


def sparse_relu(x, lam):
    return relu(x - lam)
