"""
Write functions to import memristor data.
Written by Ryan Zarcone, 2017
"""

import numpy as np
import pandas as pd
import utils.blahut as blahut

def get_raw_data(path):
    """
    Load raw data from a path.

    Parameters
    ----------
    path : str
        Absolute path to the data.

    Returns
    -------
    Vs : np.array, shape (len(data),)

    Rs : np.array, shape (len(data),)
    """
    data = pd.read_pickle(path)
    Vs = np.array(data['V'])
    Rs = np.array(data['R'])
    return Vs,Rs

def range_extender(Vs, Rs, num_ext):
    """
    Extends the range of the memristor function so that the min and max R values are
    repeated num_ext times past min and max V

    Parameters
    ----------
    Vs : np.array, shape (len(data),)

    Rs : np.array, shape (len(data),)

    num_ext : float
        Number of times the end values should be repeated
    """
    num_ext = int(num_ext)
    Vs = np.array(Vs)
    Rs = np.array(Rs)
    delta_V = Vs[1]-Vs[0]
    orig_min_Vs = np.amin(Vs)
    orig_max_Vs = np.amax(Vs)
    for i in range(num_ext):
        min_Vs = np.amin(Vs)
        max_Vs = np.amax(Vs)
        min_Vs_indx = [Vs == min_Vs]
        max_Vs_indx = [Vs == max_Vs]
        Rs_min = Rs[min_Vs_indx]
        Rs_max = Rs[max_Vs_indx]
        Vs_min = Vs[min_Vs_indx] - delta_V
        Vs_max = Vs[max_Vs_indx] + delta_V
        Vs = np.append(Vs,Vs_min)
        Vs = np.append(Vs,Vs_max)
        Rs = np.append(Rs,Rs_min)
        Rs = np.append(Rs,Rs_max)
    return Vs, Rs, orig_min_Vs, orig_max_Vs

def normalizer(x,new_min,new_max):
    x_max = np.amax(x)
    x_min = np.amin(x)

    return (((x-x_min)/(x_max-x_min))*(new_max-new_min)+new_min)

def get_pcm_data(path, n_mem, num_ext=5, norm_min=-1., norm_max=1.):
    """

    Parameters
    ----------
    path : str
        Absolute path to the data.

    n_mem : float
        Number of memristors we want to simulate

    num_ext : float
        Number of times the end values should be repeated (see range_extender)

    Returns
    -------
    vs : np.array, shape (n_samp, n_mem)

    mus : np.array, shape (n_samp, n_mem)

    sigs : np.array, shape (n_samp, n_mem)

    orig_min_Vs : float

    orig_max_Vs : float
    """
    Vs,Rs = get_raw_data(path)
    Rs = np.log10(Rs)
    Vs = np.array(Vs)
    Rs = np.array(Rs)

    orig_min_Vs = np.amin(Vs)
    orig_max_Vs = np.amax(Vs)
    orig_min_Rs = np.amin(Rs)
    orig_max_Rs = np.amax(Rs)

    Vs = normalizer(Vs,norm_min,norm_max)
    Rs = normalizer(Rs,norm_min,norm_max)

    Vs, Rs, _, _ = range_extender(Vs,Rs,num_ext)

    mus, sigs, vs = blahut.moments(Vs,Rs)

    vs = np.broadcast_to(vs[:,None], (vs.size, n_mem)).astype(np.float32)
    mus = np.broadcast_to(mus[:,None], (mus.size, n_mem)).astype(np.float32)
    sigs = np.broadcast_to(sigs[:,None], (sigs.size, n_mem)).astype(np.float32)

    return vs, mus, sigs, orig_min_Vs, orig_max_Vs, orig_min_Rs, orig_max_Rs

def get_gauss_data(path, n_mem, num_ext=5, norm_min=-1., norm_max=1.):
    """
    Simulates some memristors.

    """
    Vs,_ = get_raw_data(path)
    Vs = np.array(Vs)
    Vs = np.repeat(Vs,10)
    eta = np.random.normal(0,0.3,len(Vs)) #0.085 for 2.68 bits
    Rs = Vs + eta
    
    orig_min_Vs = np.amin(Vs)
    orig_max_Vs = np.amax(Vs)
    orig_min_Rs = np.amin(Rs)
    orig_max_Rs = np.amax(Rs)

    Vs = normalizer(Vs,norm_min,norm_max)
    Rs = normalizer(Rs,norm_min,norm_max)

    Vs, Rs, _, _ = range_extender(Vs,Rs,num_ext)

    mus, sigs, vs = blahut.moments(Vs,Rs)

    vs = np.broadcast_to(vs[:,None], (vs.size, n_mem)).astype(np.float32)
    mus = np.broadcast_to(mus[:,None], (mus.size, n_mem)).astype(np.float32)
    sigs = np.broadcast_to(sigs[:,None], (sigs.size, n_mem)).astype(np.float32)

    return vs, mus, sigs, orig_min_Vs, orig_max_Vs, orig_min_Rs, orig_max_Rs

def get_rram_data(path, n_mem, num_ext=5, norm_min=-1., norm_max=1.):
    """
    Simulates rram array.

    """
    Vs,_ = get_raw_data(path)
    Vs = np.array(Vs)
    
    ## for RRAM device with read/verify scheme ##
    b = np.log10(np.sqrt(2.0))
    Vs = normalizer(Vs, 4, 8) #RRAM goes from 4 to 8 in R_Target (log scale, so actually 10^4, 10^8)
    #################################################
    Vs = np.repeat(Vs,10)
    eta = np.random.normal(0,b**2,len(Vs))
    Rs = Vs + eta
    
    orig_min_Vs = np.amin(Vs)
    orig_max_Vs = np.amax(Vs)
    orig_min_Rs = np.amin(Rs)
    orig_max_Rs = np.amax(Rs)

    Vs = normalizer(Vs,norm_min,norm_max)
    Rs = normalizer(Rs,norm_min,norm_max)

    Vs, Rs, _, _ = range_extender(Vs,Rs,num_ext)

    mus, sigs, vs = blahut.moments(Vs,Rs)

    vs = np.broadcast_to(vs[:,None], (vs.size, n_mem)).astype(np.float32)
    mus = np.broadcast_to(mus[:,None], (mus.size, n_mem)).astype(np.float32)
    sigs = np.broadcast_to(sigs[:,None], (sigs.size, n_mem)).astype(np.float32)

    return vs, mus, sigs, orig_min_Vs, orig_max_Vs, orig_min_Rs, orig_max_Rs


def make_piecewise():
    # Check channel capacity of artificial memristor.
    pass
