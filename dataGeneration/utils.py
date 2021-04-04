import os
import torch
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from collections import defaultdict
from collections.abc import Iterable

from utils.parameters import EPS_0
from utils.operations import makeList


def getP(r1, r2):
    '''
        Returns P from r1 & r2
        (P is the ratio of the shell volume to the total particle volume)
    '''
    return 1-(r1/r2)**3


def getEps(e1, e2, params, mode='P'):
    '''
        mode : 
            'P' => given P => params = P
            'R' => given r1 & r2 in params => params = (r1, r2)
    '''
    if mode=='R':
        r1, r2 = params
        P = getP(r1, r2)
    else:
        P = params
    
    # For multiple r1 & r2 values, we'll have to do matrix multiplication
    # as P will be an array as well
    if isinstance(P, Iterable):
        if torch.is_tensor(e1):
            if P.ndim==0:
                ea = e1*(3-2*P) + 2*e2*P
                eb = e1*P + e2*(3-P)
            else:
                ea = torch.outer(e1, (3-2*P)) + 2*torch.outer(e2, P)
                eb = torch.outer(e1, P) + torch.outer(e2, 3-P)
        else:
            ea = np.outer(e1, (3-2*P)) + 2*np.outer(e2, P)
            eb = np.outer(e1, P) + np.outer(e2, 3-P)

    # For a single (r1, r2) sample pair, ea & eb will be same as e1 & e2
    else:
        ea = e1*(3-2*P) + 2*e2*P
        eb = e1*P + e2*(3-P)

    return ea, eb


def getAlpha(eps, r, baked=False):
    '''
        Returns alpha from epsilon and radius values
        Args:
            baked : Flag that defines variable types
            eps : dict containing multiple keys of different epsilons
                e1 (baked=False)
                e2
                e3
                ea (baked=True)
                eb (baked=True)
            r : dict containing multiple keys of different radii
                r1 (baked=False)
                r2
    '''
    e2 = eps['e2']
    e3 = eps['e3']
    r2 = r['r2']

    if baked:
        ea = eps['ea']
        eb = eps['eb']
    else:
        e1 = eps['e1']
        r1 = r['r1']
        ea, eb = getEps(e1, e2, params=(r1, r2), mode='R')
    
    # For multiple (r1, r2) pairs, ea & eb will be 2D
    # In that case to execute following calculations, we'll
    # have to unsqueeze e1 & e2 as 2D vectors
    if ea.ndim == 2:
        if torch.is_tensor(e1):
            e1 = torch.unsqueeze(e1, dim=1)
            e2 = torch.unsqueeze(e2, dim=1)
            if e3.ndim == 1:
                e3 = torch.unsqueeze(e3, dim=1)
        else:
            e1 = np.expand_dims(e1, axis=1)
            e2 = np.expand_dims(e2, axis=1)

    alpha = 4*np.pi*EPS_0*(r2**3) * (e2*ea-e3*eb)/(e2*ea+2*e3*eb)
    return alpha


def getArea(r, eps, lambd, write=None, f_out=None, ret='default'):
    '''
        Returns area from radius & epsilon input
        Args: [All args are in SI unit]
            r : radius data (dict)
                r1
                r2
            eps : epsilon data (dict)
                e1
                e2
                e3
            lambd : wavelength
            write : To write generated data or not
                    None/False => don't write
                    'all' => write all indermediate variables
                    ['var1', 'var2', .....] => write only mentioned variables
            f_out : write file name (with location)
                    [only valid when write!=None and write!=False]
        Returns:
            area : float
    '''
    k = 2*np.pi/lambd
    alpha = getAlpha(eps, r, baked=False)

    # Scattering cross section
    if alpha.ndim == 2:
        if torch.is_tensor(k):
            k = torch.unsqueeze(k, dim=1)
        else:
            k = np.expand_dims(k, axis=1)
    area_sca = (1/(6*np.pi*(EPS_0**2))) * (k**4) * (abs(alpha)**2)
    
    # Absorption cross section
    area_abs = k * alpha.imag / EPS_0

    if (write!=False) and (write is not None):

        if f_out==None:
            raise AssertionError("Please provide an export filename for writing data or set write to None")

        if not os.path.exists(os.path.dirname(f_out)):
            os.makedirs(os.path.dirname(f_out))

        # If write='all', put all variables in it
        if write=='all':
            write = ['alpha', 'A_sca', 'A_abs', 'A_tot']
        
        n = len(lambd)
        data = {
            'lambd': lambd,
            'r1': makeList(r['r1'], n),
            'r2': makeList(r['r2'], n),
            'eps_1': makeList(eps['e1'], n),
            'eps_2': makeList(eps['e2'], n),
            'eps_3': makeList(eps['e3'], n),
        }

        for w in write:
            if w=='alpha':
                data[w] = alpha
            elif w=='A_sca':
                data[w] = area_sca
            elif w=='A_abs':
                data[w] = area_abs
            elif w=='A_tot':
                data[w] = area_sca + area_abs
        
        data = pd.DataFrame(data)
        data.to_csv(f_out, index=False)
    
    if ret=='default':
        return area_sca, area_abs
    elif ret=='A_abs' or ret=='abs':
        return area_abs
    elif ret=='A_sca' or ret=='sca':
        return area_sca
    else:
        return TypeError(f"Unknown return type - {ret} found")

