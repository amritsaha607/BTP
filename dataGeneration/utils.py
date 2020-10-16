import numpy as np
from utils.parameters import EPS_0



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
    
    alpha = 4*np.pi*EPS_0*(r2**3) * (e2*ea-e3*eb)/(e2*ea+2*e3*eb)
    return alpha


def getArea(r, eps, lambd):
    '''
        Returns area from radius & epsilon input
        Args:
            r : radius data (dict)
                r1
                r2
            eps : epsilon data (dict)
                e1
                e2
                e3
            lambd : wavelength
        Returns:
            area : float
    '''
    k = 2*np.pi/lambd
    alpha = getAlpha(eps, r, baked=False)
    area = (1/(6*np.pi*(EPS_0**2))) * (k**4) * (alpha**2)
    return area

