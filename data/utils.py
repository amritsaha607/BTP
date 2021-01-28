import os
import numpy as np
import pandas as pd
from torch._C import Value


def extractData(filename, 
            input_key='A_tot',
            factors={'r': 1e9, 'eps': 1e7, 'lambd': 1e9, 'A': 1e17}):
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

    # Sampled values of cross section at specified lambd interval
    x = f_A*df[input_key].values

    # First all wavelength data followed by area data
    # x = np.concatenate([f_lambd*df['lambd'].values, f_A*df['A_tot'].values], axis=0)

    # Output contains parameters like r1, r2, eps_1, eps_2 & eps_3
    y = np.array([
        f_r*df['r1'][0],
        f_r*df['r2'][0],
        f_eps*complex(df['eps_1'][0]).real,
        f_eps*complex(df['eps_1'][0]).imag,
        f_eps*complex(df['eps_3'][0]).real,
        f_eps*complex(df['eps_3'][0]).imag
    ])

    mapping = np.vectorize(lambda t: complex(t.replace('i', 'j')))
    eps_2 = df['eps_2'].values
    eps_2 = mapping(eps_2)
    y = np.concatenate([y, f_eps*(eps_2.real), f_eps*(eps_2.imag)], axis=0)

    return x, y



class PermittivityCalculator:
    """
        Calculates permittivity for elements as a function of wavelength
    """
    def __init__(self):
        pass

    def getEps(self, lambd,
        element="sio2"):
        """
            Calculates permittivity from wavelength 
            [Reference : https://refractiveindex.info/]
            Args:
                lambd : wavelength
                element : element name
        """
        if element == "sio2":
            # https://refractiveindex.info/?shelf=main&book=SiO2&page=Radhakrishnan-o
            n_sq = 1 + 0.663044*(lambd**2)/(lambd**2-0.060**2) + 
                0.517852*(lambd**2)/(lambd**2-0.106**2) + 0.175912*(lambd**2)/(lambd**2-0.119**2) + 
                0.565380*(lambd**2)/(lambd**2-8.844**2) + 1.675299*(lambd**2)/(lambd**2-20.742**2)
            k_sq = 0.0

        elif element == "al2o3":
            # https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-o
            n_sq = 1 + 1.4313493*(lambd**2)/(lambd**2-(0.0726631**2)) + 0.65054713*(lambd**2)/(lambd**2-(0.1193242**2)) + 5.3414021*(lambd**2)/(lambd**2-(18.028251**2))
            k_sq = 0.0

        elif element == "hgs":
            # https://refractiveindex.info/?shelf=main&book=HgS&page=Bond-o
            n_sq = 1 + 3.1506 + 2.7896*(lambd**2)/(lambd**2-0.1328) + 
                1.1378*(lambd**2)/(lambd**2-1705)
            k_sq = 0.0

        elif element == "aggas2":
            # https://refractiveindex.info/?shelf=main&book=AgGaS2&page=Takaoka-o
            n_sq = 5.7975 + 0.2311/(lambd**2-0.0688) -
                0.00257*(lambd**2)
            k_sq = 0.0
        
        elif element == "batio3":
            # https://refractiveindex.info/?shelf=main&book=BaTiO3&page=Wemple-o
            n_sq = 1 + 4.187*(lambd**2)/(lambd**2-(0.223**2)) 
            k_sq = 0.0

        elif element == "tio2":
            # https://refractiveindex.info/?shelf=main&book=TiO2&page=Bodurov
            n_sq = 1 + 4.6796*(lambd**2)/(lambd**2-(0.2002148**2)) 
            k_sq = 0.0

        else:
            raise ValueError("Unknown element found - {}".format(element))
        
        eps_r = n_sq - k_sq
        eps_i = 2*(n_sq**0.5)*(k_sq**0.5)

        return (eps_r, eps_i)
