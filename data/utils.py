import os
import numpy as np
import pandas as pd


def extractData(filename):
    '''
        Extracts data from a csv file for training
        Args:
            filename    : filename to extract data from
                        [supported formats => '.csv']
    '''

    df = pd.read_csv(filename)

    # Input contains two columns, wavelength and area
    # x = np.c_[df['lambd'], df['A_tot']]

    # Input data is a list of combined wavelength and area
    # First all wavelength data followed by area data
    x = np.concatenate([df['lambd'].values, df['A_tot'].values], axis=0)

    # Output contains parameters like r1, r2, eps_1, eps_2 & eps_3
    y = np.array([
        df['r1'][0],
        df['r2'][0],
        complex(df['eps_1'][0]).real,
        complex(df['eps_1'][0]).imag,
        complex(df['eps_2'][0]).real,
        complex(df['eps_2'][0]).imag,
        complex(df['eps_3'][0]).real,
        complex(df['eps_3'][0]).imag
    ])

    return x, y