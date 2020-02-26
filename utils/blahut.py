import operator
import numpy as np
from scipy import stats, interpolate

#==============================================================================
# This library module is full of functions and classes to compute the maximum
# mutual information (Capacity) between an input (x) (voltage) and output (y) (resistance)
# distribution transmitted through a noisy channel (Pyx) (device).
# Free to use and distribute and alter.
# Created by Jesse Engel (Stanford, UC Berkeley) Sept 2, 2014
#==============================================================================



#==============================================================================
# Discrete Distribution Functions
#==============================================================================
def h(p):
    """Shannon Information
    """
    info = -1*np.log2(p)
    if np.isscalar(info):
        if np.isinf(info) or info == -0.:
            info = 0
    else:
        info[np.where(info == -0.)] = 0
        info[np.where(np.isinf(info))] = 0
    return info

def H(p):
    """Entropy
    """
    return p * h(p)


def H2(p):
    """Binary Entropy
    """
    return H(p) + H(1-p)


def D_KL(p, q):
    '''
    Compute the KL Diveregence of two finite distributions p and q

    Params
    ------
    p
        (array) [np]
    q
        (array) [nq]

    Returns
    -------
    D_KL
        (float) in Bits
    '''
    if p.ndim == 2:
        #D_KL for each row of p
        d = p * np.log2(p / np.r_[q])
        d[np.logical_not(np.isfinite(d))] = 0
        return np.sum(d,1)
    else:
        d = p * np.log2(p / q)
        d[np.logical_not(np.isfinite(d))] = 0
        return np.sum(d)


def I(Pyx, Px):
    '''
    Compute the mutual information of distribution Px traveling through
    channel Pyx.

    Params
    ------
    Pyx
        (array) [ny, nx]
    Px
        (array) [nx]

    Returns
    -------
    I
        (float) in Bits
    '''
    Pyx = Pyx.T
    Py = np.dot(Px, Pyx)
    I = np.dot(Px, D_KL(Pyx, Py))
    return I

#==============================================================================
# Vectorized Blahut-Arimoto algorithm
#==============================================================================

def blahut_arimoto(Pyx, tolerance = 1e-2, iterations = 1000,
                   e=np.empty(0), s=0, debug=False, Px0=np.empty(0)):
    '''
    Blahut-Arimoto algorithm for computing the Capacity of a discrete input-output channel
    Based on a matlab code by: Kenneth Shum
    http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=825

    Adapted from Blahut 1972, IEEE Trans. on Info. Theory

    Params
    ----

    Pyx
        Discrete conditional probability matrix.
        (array) [ny, nx]

    Keywords
    -----
    e
        Vector of expenses for given x input states
    s
        Lagrange multiplier. One to one mapping to an average
    tolerance
        End when IU - IL < tolerance
    iterations
        Max number of iterations
    debug:
        Print to console while running

    Outputs
    ----

    C
        (float) Capacity in bits
    Px
        (array) [nx] Optimal input distribution
    E
        (float) Expense. Only output if 'e' is defined

    '''
    Pyx = Pyx.T # (inputs, outputs)


    m, n = Pyx.shape # (m inputs, n outputs)
    Px = [np.ones(m)/m, Px0][Px0.any()] # initial distribution for channel input
    Py = np.ones(n)/n # initial distribution for channel output
    c = np.zeros(m)
    energy_constraint = e.any()
    D = D_KL(Pyx, Py) #Vector


    temp = Pyx / np.c_[np.sum(Pyx,1)]
    ind = np.isfinite(temp)
    Pyx[ind] = temp[ind]


    #Optimizaiton
    for i_iter in np.arange(iterations):

        if energy_constraint:
            c = np.exp( D - s*e )
        else:
            c = np.exp( D )

        #Update
        Px = Px * c
        Px = Px/np.sum(Px)
        Py = np.dot(Px, Pyx)
        D = D_KL(Pyx, Py) #Vector

        IL = np.log(np.dot(Px, c))
        IU = np.log(max(c))


        if debug:
            if energy_constraint:
                E = np.dot(Px, e)
                print ('\nE: %.2e' % E)
                print ('IL: %.2e IU: %.2e' % (IL, IU))
                print ('Iter: %d' % (i_iter+1))

            else:
                print ('\nIL: %.2e IU: %.2e' % (IL, IU))
                print ('Iter: %d' % (i_iter+1))

        if tolerance:
            if IU-IL < tolerance:
                break


    C = I(Pyx.T, Px)

    if debug:
        print ('iterations: %s' % (i_iter+1))
        print ('C:', C)


    if energy_constraint:
        E = np.dot(Px, e)
        return C, Px, E
    else:
        return C, Px





# def rate_distortion()
# Rate-Distortion is for SOURCE compression.
# It calculates the lower bound of required description length to compress and
# reconstruct a GIVEN source (Px (can be multidimensional and dependent, like in images)).
# It does NOT tell you how to achieve that compression 'codebook and code points', except
# for simple cases like independent (iid) gaussian sources. In that case it actually works out that
# doing multidimensional to single dimensional compression (vector quantization) is better than scalar quantization

# Problems of communication theory:
# ------------
# 1) WHAT information should be transmitted? (source coding)
# 2) HOW should it be transmitted? (channel coding)

# These two problems can be separated by Shannon's separation theorem, and the distortion
# will never exceed D(R) as long as R < C.

# But what of joint coding?


#==============================================================================
# Quantization of high-D quantized channel to low-D quantized channel
#==============================================================================

def find_closest(vector, value):
    ''' Find the closest index of a vector to a value. If value is a vector,
    returns an array of indicies that are closest to the values. Rounds down.
    '''
    if isinstance(value, np.ndarray) or isinstance(value, list):
        diff = np.outer(vector, np.ones(len(value))) - np.outer(np.ones(len(vector)), value)
        inds = np.argmin( np.abs(diff), axis=0)
    else:
        inds = np.argmin( np.abs( vector - value) )
    return inds



def calc_subsample_inds(x, y, xinputs=None, ydividers=None):
    ''' Find closest indexes for a discetization of x and y
    '''
    if np.any(xinputs):
        xinds = find_closest(x, xinputs)
    else:
        xinds = np.arange(x.size)
    if np.any(ydividers):
        yinds = find_closest(y, ydividers)
    else:
        yinds = np.arange(y.size)

    return xinds, yinds


def subsample(Pyx, xinds, yinds):
    ''' Subsample a density matrix a locations xinds(columns), and sum between
    dividers yinds(row).
    '''
    Pyx_sub = np.zeros( [len(yinds)+1, len(xinds)])
    bounds = np.r_[0, yinds, Pyx.shape[0]]


    for i in np.arange(len(bounds)-1):
        bl = bounds[i]
        bu = bounds[i+1]
        Pyx_sub[i,:] = np.sum(Pyx[bl:bu, xinds], axis = 0)

    # Normalize
    Pyx_sub = Pyx_sub / np.sum(Pyx_sub, axis=0)

    return Pyx_sub


def quantize(Pyx, x, y, xinputs=None, ydividers=None):
    '''Chops up a matrix Pyx, into xinputs columns, and sums rows between
    y dividers
    '''
    xinds, yinds = calc_subsample_inds(x, y, xinputs, ydividers)
    Pyx_sub = subsample(Pyx, xinds, yinds)
    x_sub = x[xinds]
    y_sub = y[::-1][yinds]
    return Pyx_sub, x_sub, y_sub



def trim(Pyx, cum_low=1e-1, cum_high=1e-1, index=False):
    '''Returns Pyx only rows where cumsum(P) > cum_low and cumsum(P) < 1 - cum_high'''
    low =  min( np.where( np.cumsum(Pyx, axis=0) > cum_low )[0])
    high = max( np.where( np.cumsum(Pyx, axis=0) < 1-cum_high )[0])
    if not index:
        return Pyx[low:high, :]
    else:
        return Pyx[low:high, :], np.arange(low, high+1)




#==============================================================================
# Gaussian Kernel Density Estimate from Data
#==============================================================================
def Q(Varray, Rarray, nx=2000, ny=2000,print_points=True):
    '''
    Take in all Voltage/Resistance Pairs and return the conditional PDF: Q= P(R|V)
    Performs Gaussian Kernel Density Estimate and Linear Interpolation

    Params
    ------
    Varray, Rarray
        ndarray, same size (n_examples,)

    Returns
    -------
    Q
        ndarray (__, __)
    '''
    V_list = np.sort(np.unique(Varray))

    #Gaussian KDE
    Pyx_func = []


    for i, v in enumerate(V_list):
        idx = (Varray == v)
        data = Rarray[idx]
        if print_points == True:
            print ('%0.2f Volts, %d Points' % (v, sum(idx)))
        Pyx_func.append(stats.gaussian_kde(data, bw_method='scott' )) #scott, silvermann, scalar


    Pyx_func = FunctionList(Pyx_func)

    x = np.linspace(V_list.min(), V_list.max(), nx)
    y = np.linspace(Rarray.min()*0.7, Rarray.max()*1.3, ny)

    # Bivariate Spline
    Pyx = np.atleast_2d(Pyx_func(y))
    Pyx_interp = interpolate.RectBivariateSpline( V_list, y, Pyx, kx=3, ky=3, s=0)
    Pyx_new = np.rot90(Pyx_interp(x,y))

    # Normalize (each input needs to end up in an output (column=1))
    Pyx_new = Pyx_new / np.sum(Pyx_new, axis=0)

    return Pyx_new, x, y


def moments(Varray, Rarray):
    '''
    Returns mean, variance of a R(V) dataset
    '''
    V_list = np.sort(np.unique(Varray))
    data_mean = np.zeros(V_list.size)
    data_var = np.zeros(V_list.size)
    Vs = np.zeros(V_list.size)

    for i, v in enumerate(V_list):
        idx = (Varray == v)
        data = Rarray[idx]
        data_mean[i] = np.mean(data)
        data_var[i] = np.var(data)
        Vs[i] = v

    return data_mean, data_var, Vs

#==============================================================================
# Classes
#==============================================================================
class FunctionList(object):
    def __init__(self, f_list):
        """
        FunctionList is a list of function objects that can be
        added, multiplied, summed, and dot producted with ints/floats,
        functions, np.array()s, and other FunctionLists.

        This is a bit of a hack to allow for making an array of functions.

        Parameters
        ----------
        f_list : list of functions

        Examples
        --------
        >>> f = lambda x: x
        >>> g = FunctionList([f, f])
        >>> h=g.dot([1,2])
        >>> g(2)
        [2, 2]
        >>> h(2)
        6
        """
        if type(f_list) is FunctionList:
            self.f_list = f_list.f_list
        elif hasattr(f_list, '__call__'):
            self.f_list = [f_list]
        else:
            self.f_list = f_list

    def __call__(self, x):
        result = []
        for f in self.f_list:
            result.append( f(x) )
        return result

    def __add__(self, other):
        """ Add the function list, elementwise: Returns a function list
        """
        return self.__apply_op(other, op=operator.add)

    def __sub__(self, other):
        """ Add the function list, elementwise: Returns a function list
        """
        return self.__apply_op(other, op=operator.sub)

    def __mul__(self, other):
        """ Multiply the function list, elementwise: Returns a function list
        """
        return self.__apply_op(other, op=operator.mul)

    def __div__(self, other):
        """ Divide the function list, elementwise: Returns a function list
        """
        return self.__apply_op(other, op=operator.div)

    def __apply_op(self, other, op=operator.add):
        result = []

        if type(other) is FunctionList:
            for i, f in enumerate(self.f_list):
                g = other[i]
                result.append( lambda x, f=f, g=g: op(f(x), g(x)) )

        elif hasattr(other, '__call__'):
            for f in self.f_list:
                g = other
                result.append( lambda x, f=f, g=g: op(f(x), g(x)) )

        elif type(other) in (np.ndarray, list):
            for i, f in enumerate(self.f_list):
                g = other[i]
                result.append( lambda x, f=f, g=g: op(f(x), g) )

        elif type(other) in (int, float):
            for f in self.f_list:
                g = other
                result.append( lambda x, f=f, g=g: op(f(x), other) )

        else:
            print ('Add FunctionList with: FunctionList, ndarray, int, or float')
            pass
        return FunctionList(result)

    def sum(self):
        result = self.f_list[0]
        for i, g in enumerate(self.f_list[1:]):
            f = result
            result = lambda x, f=f, g=g: f(x) + g(x)
        return result

    def dot(self, other):
        """Take the dot product of a function vector and either another
        function vector, or a normal vector.
        """
        result = self.__mul__(other)
        result = result.sum()
        return result


    def __getitem__(self,index):
        return self.f_list[index]

    def __setitem__(self,index,value):
        self.f_list[index] = value

    def __len__(self):
        return len(self.f_list)
