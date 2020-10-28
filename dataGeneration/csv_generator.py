import pandas as pd
import numpy as np
from collections import defaultdict

def getEps(f_in='csv/Au.csv', f_out='csv/Au_updated.csv'):
    '''
        Takes input CSV file containing wavelength, n & k as columns
        refractive index = n+j*k
        epsilon = (refractive index)^2
        Calculates real and imaginary part of epsilon and writes new csv
    '''

    content = pd.read_csv(f_in)
    data = defaultdict(list)

    for (wl, n, k) in zip(content['wl'], content['n'], content['k']):
        er = n**2-k**2
        ei = 2*n*k
        data['wl'].append(wl*1e3)   # um => nm
        data['n'].append(n)
        data['k'].append(k)
        data['er'].append(er)
        data['ei'].append(ei)
    
    data = pd.DataFrame(data)
    data.to_csv(f_out, index=False) 
    

def getEpsInterpolated(f_in='csv/Au.csv', f_out='csv/Au_interpolated.csv', range_='auto', interval=1):
    '''
        Takes input CSV file containing wavelength, n & k as columns
        Interpolated the data with interval of "interval" w.r.t. wavelength
        refractive index = n+j*k
        epsilon = (refractive index)^2
        Calculates real and imaginary part of epsilon and writes new csv
    '''

    content = pd.read_csv(f_in)
    data = defaultdict(list)

    wls, ns, ks = content['wl']*1e3, content['n'], content['k']

    if range_=='auto':
        range_ = [wls.min(), wls.max()]

    wls_all = np.arange(range_[0], range_[1], interval)
    # wls_all = np.arange(wls.min(), wls.max(), interval)
    data['wl'] = wls_all
    data['n'] = np.interp(wls_all, wls, ns)
    data['k'] = np.interp(wls_all, wls, ks)
    data['er'] = data['n']**2 - data['k']**2
    data['ei'] = 2 * data['n'] * data['k']
    
    data = pd.DataFrame(data)
    data.to_csv(f_out, index=False) 
    
