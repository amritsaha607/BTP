import os
import numpy as np
import pandas as pd


def extractData(filename, factors={'r': 1e9, 'eps': 1e7, 'lambd': 1e9, 'A': 1e17}):
    '''
        Extracts data from a csv file for training
        Args:
            filename    : filename to extract data from
                        [supported formats => '.csv']
    '''

    if not factors:
        f_r, f_eps, f_lambd, f_A = 1, 1, 1, 1
    else:
        f_r, f_eps, f_lambd, f_A = factors['r'], factors['eps'], factors['lambd'], factors['A']
    
    df = pd.read_csv(filename)

    # Input contains two columns, wavelength and area
    # x = np.c_[df['lambd'], df['A_tot']]

    # Input data is a list of combined wavelength and area
    # First all wavelength data followed by area data
    x = np.concatenate([f_lambd*df['lambd'].values, f_A*df['A_tot'].values], axis=0)

    # Output contains parameters like r1, r2, eps_1, eps_2 & eps_3
    y = np.array([
        f_r*df['r1'][0],
        f_r*df['r2'][0],
        f_eps*complex(df['eps_1'][0]).real,
        f_eps*complex(df['eps_1'][0]).imag,
        f_eps*complex(df['eps_2'][0]).real,
        f_eps*complex(df['eps_2'][0]).imag,
        f_eps*complex(df['eps_3'][0]).real,
        f_eps*complex(df['eps_3'][0]).imag
    ])

    return x, y