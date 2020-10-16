import pandas as pd
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
        data['wl'].append(wl)
        data['n'].append(n)
        data['k'].append(k)
        data['er'].append(er)
        data['ei'].append(ei)
    
    data = pd.DataFrame(data)
    data.to_csv(f_out, index=False) 
    
