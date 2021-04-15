from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import os
import xlsxwriter
import openpyxl
import matplotlib

from dataGeneration.utils import getArea


def oneHot(index, n):
    """
        Creates one hot vector of n classes with "index" class
    """
    x = np.zeros(n)
    x[index] = 1
    return x


def isMode(mode, check):
    """
        Check if mode contains "check"
    """
    if mode=="default" or mode=="all":
        return True
    
    if mode.__contains__(check):
        return True

    if check.__contains__("_"):
        check_modes = check.split("_")
        for check_mode in check_modes:
            if not isMode(mode, check_mode):
                return False
        return True

    return False


def getLabel(y, mode='default'):
    """
        Extracts needed labels from given data of all labels
        Args:
            y   : all labels
            mode:
                default: all output
                r: radius
                eps_sm: only e1 & e3 (both real and imaginary parts)
                eps: epsilon data
    """
    if mode=='default':
        return y
    elif mode=='r':
        y = y[:, :2]
        return y
    elif mode=="eps_sm":
        y = y[:, 2:6]
        return y
    elif mode=='eps':
        y = y[:, 2:]
        return y
    else:
        raise ValueError("Unknown mode {} found".format(mode))


def getPredictions(y):

    '''
        Returns prediction dict from raw model output
        Args:
            y : model output
                shape : (batch_size X (2n+6))
    '''
    
    batch_size, tot_dim = y.shape[0], y.shape[1]
    n = (tot_dim - 6) // 2
    r1 = y[:, 0:1]
    r2 = y[:, 1:2]
    e1 = y[:, 2:4]
    e3 = y[:, 4:6]
    e2_r = y[:, 6:6+n]
    e2_i = y[:, 6+n:]
    res = {
        'r1': r1,
        'r2': r2,
        'e1': e1,
        'e3': e3,
        'e2_r': e2_r,
        'e2_i': e2_i
    }
    return res


def getAreaE1Class(r, e1_cls, 
    lambd=None,
    e1=None,
    eps=None,
    data_file='dataGeneration/csv/Au_interpolated_1.csv',
    data_factors=None,
    ret_mode='all'):

    """
        Returns cross sections from given r data & e1_cls
        Args:
            r : 
                (dict) : returns single sample area
                (list of list) : returns multiple samples area as a list of np array
            e1_cls : string [material name of e1 class]
            lambd : wavelength array
            e1 : e1 array
            eps : eps dict [e1 => array, e2 => array, e3 => complex]
            data_file : file to read annotations from
            data_factors : factors to apply in cross sectional data
            ret_mode : 
                "all" : returns [A_sca, A_abs]
                "abs" : returns A_abs
                "sca" : returns A_sca
    """

    if isinstance(r, list):
        x = []
        n_r = len(r)

        # If eps is not calculated let's calculate it
        # so that we don't have to calculate it everytime in loop
        if not eps:
            e3 = 1.00 + 1j*0.0

            if lambd is None:
                content = pd.read_csv(data_file)
                lambd = 1e-9 * content['wl'].values
                e2 = content['er'].values + 1j*content['ei'].values

            if e1 is None:
                from data.utils import PermittivityCalculator
                pc = PermittivityCalculator()
                e1 = np.array([pc.getEps(wl_, element=e1_cls, mode="complex") for wl_ in lambd*1e6])

            eps = {
                'e1': e1,
                'e2': e2,
                'e3': e3,
            }

        for i in range(n_r):
            tr = {
                'r1': r[i][0],
                'r2': r[i][1],
            }
            A_sca, A_abs = getAreaE1Class(tr, e1_cls, lambd=lambd, eps=eps, data_factors=data_factors)

            if ret_mode == 'abs':
                x.append(A_abs)
            elif ret_mode == 'sca':
                x.append(A_sca)
            else:
                x.append([A_sca, A_abs])

        return x

    if data_factors is None:
        data_factors = defaultdict(lambda: 1)

    if not eps:

        if lambd is None:
            content = pd.read_csv(data_file)
            lambd = 1e-9 * content['wl'].values
            e2 = content['er'].values + 1j*content['ei'].values

        if e1 is None:
            from data.utils import PermittivityCalculator
            pc = PermittivityCalculator()
            e1 = np.array([pc.getEps(wl_, element=e1_cls, mode="complex") for wl_ in lambd*1e6])

        eps = {
            'e1': e1,
            'e2': e2,
            'e3': 1.0 + 1j*0.0,
        }

    A_sca, A_abs = getArea(r, eps, lambd)
    # if r['r1'] == 10e-9 and r['r2'] == 20e-9:
    #     print(e1_cls)
    #     print(data_factors)
    #     print(A_abs[:5])
    #     print(A_abs.min(), A_abs.max())
    #     p
    A_sca, A_abs = A_sca*data_factors['A'], A_abs*data_factors['A']

    return A_sca, A_abs


def getAreaE1E2Class(r, e1_cls, e2_cls,
    lambd=None,
    e1=None,
    e2=None,
    eps=None,
    data_file='dataGeneration/csv/Au_interpolated_1.csv',
    data_factors=None,
    ret_mode='all'):

    """
        Returns cross sections from given r data, e1_cls & e2_cls
        Args:
            r : (raw data, given in nanometers)
                (dict) : returns single sample area
                (list of list) : returns multiple samples area as a list of np array
            e1_cls : string [material name of e1 class]
            e2_cls : string [material name of e2 class]
            lambd : wavelength array
            e1 : e1 array
            e2 : e2 array
            eps : eps dict [e1 => array, e2 => array, e3 => complex]
            data_file : file to read annotations from
            data_factors : factors to apply in cross sectional data
            ret_mode : 
                "all" : returns [A_sca, A_abs]
                "abs" : returns A_abs
                "sca" : returns A_sca
    """

    if data_factors is None:
        data_factors = defaultdict(lambda: 1)

    multi = False
    if isinstance(r, list):
        r = np.array(r)
        r = {
            'r1': r[:, 0],
            'r2': r[:, 1],
        }
        multi = True

    if not eps:

        if lambd is None:
            content = pd.read_csv(data_file)
            lambd = 1e-9 * content['wl'].values
            e2 = content['er'].values + 1j*content['ei'].values

        if e1 is None or e2 is None:
            from data.utils import PermittivityCalculator
            pc = PermittivityCalculator()

        if e1 is None:
            e1 = np.array([pc.getEps(wl_, element=e1_cls, mode="complex") for wl_ in lambd*1e6])

        if e2 is None:
            e2 = np.array([pc.getEpsArr(wl_, element=e2_cls, mode="complex") for wl_ in lambd*1e6])

        eps = {
            'e1': e1,
            'e2': e2,
            'e3': 1.0 + 1j*0.0,
        }

    A_sca, A_abs = getArea(r, eps, lambd)
    A_sca, A_abs = A_sca*data_factors['A'], A_abs*data_factors['A']

    if multi:
        A_sca, A_abs = A_sca.T, A_abs.T

    if ret_mode == 'abs':
        return A_abs
    elif ret_mode == 'sca':
        return A_sca
    elif ret_mode == 'all':
        if multi:
            return [[A_sca_, A_abs_] for (A_sca_, A_abs_) in zip(A_sca, A_abs)]
        else:
            return [A_sca, A_abs]
    else:
        raise NameError("Unknown ret_mode found - {}".format(ret_mode))


def getLossWeights(weights_dict, n):
    """
        Get loss weights from loss_weights config dict
        Args:
            weights_dict : Contains weights of different parameters
            n : number of samples
    """

    w = torch.ones(2*n+6,)
    w[0] *= weights_dict['r1']
    w[1] *= weights_dict['r2']
    w[2] *= weights_dict['e1_r']
    w[3] *= weights_dict['e1_i']
    w[4] *= weights_dict['e3_r']
    w[5] *= weights_dict['e3_i']
    w[6:6+n] *= weights_dict['e2_r']
    w[6+n:] *= weights_dict['e2_i']
    return w


def excelDfWriter(df, filename='temp.xlsx',
    sheet_name='sheet 1', dispose=False):

    """
        Write dataframe to excel file
        Args:
            df : dataframe
            filename : output filename
            sheet_name : sheet_name
            dispose : To remove the images in the dataframe (True) or save them as files (False)
    """

    if not isinstance(df, list):
        df = [df]
        sheet_name = [sheet_name]

    workbook = openpyxl.Workbook()

    # Remove any default sheet/s
    default_sheet = workbook.get_sheet_by_name("Sheet")
    workbook.remove_sheet(default_sheet)

    # Put values in different sheets
    img_counter = 1
    for (df_, sheet_name_) in zip(df, sheet_name):
        workbook.create_sheet(sheet_name_)
        worksheet = workbook[sheet_name_]
        col = 'A'

        for key in df_.keys():
            worksheet[f'{col}1'] = key

            row = 2
            for val in df_[key]:
                if isinstance(val, matplotlib.figure.Figure):
                    val.savefig(f"temp_{img_counter}.png")
                    img = openpyxl.drawing.image.Image(f"temp_{img_counter}.png")
                    worksheet.add_image(img, f'{col}{row}')
                    img_counter += 1

                    [x_dim, y_dim] = val.get_size_inches() * val.dpi
                    worksheet.column_dimensions[col].width = x_dim//8 # 25
                    worksheet.row_dimensions[row].height = (y_dim * 3) // 4 # 150
                else:
                    worksheet[f'{col}{row}'] = val
                row += 1
            
            col = chr(ord(col) + 1)

    workbook.save(filename=filename)

    if dispose:
        for i in range(1, img_counter):
            os.remove(f"temp_{i}.png")


def transform_domain(y, domain=0, reverse_=False):

    if reverse_ == False:
        if domain == 0:
            return y
        elif domain == 1:
            # a=r1/r2, b=r2-r1 => [a, b]
            y[:, 0], y[:, 1] = y[:, 0]/y[:, 1], y[:, 1]-y[:, 0]
            return y
        else:
            raise ValueError(f"Unknown domain {domain}")
    
    else:
        if domain == 0:
            return y
        elif domain == 1:
            # [a, b]
            # r1 = a*b/(1-a), b=b/(1-a) => [r1, r2]
            y[:, 0], y[:, 1] = y[:, 0]*y[:, 1]/(1-y[:, 0]), y[:, 1]/(1-y[:, 0])
            return y
        else:
            raise ValueError(f"Unknown domain {domain}")


def getPeakInfo(A, lambd, shift=150):
    """
        Returns peak position and wavelength value where peak occurs
        Args:
            A : Cross section array (torch / numpy)
            lambd : wavelength array (torch / numpy)
            shift : amount of shift in A-lambd array to consider
    """

    lambd = lambd[shift:]
    if A.ndim == 2:
        A = A[:, shift:]
        if torch.is_tensor(A):
            max_idx = A.argmax(dim=1)
            A_max = A.max(dim=1).values
        else:
            max_idx = A.argmax(axis=1)
            A_max = A.max(axis=1)
    else:
        A = A[shift:]
        max_idx = A.argmax()
        A_max = A[max_idx]

    lambd_max = lambd[max_idx]
    return lambd_max, A_max
