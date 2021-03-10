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

    if not isinstance(df, list):
        df = [df]
        sheet_name = [sheet_name]

    workbook = openpyxl.Workbook()
    # for sheet_name_ in sheet_name:
    #     workbook.create_sheet(sheet_name_)
    
    for (df_, sheet_name_) in zip(df, sheet_name):
        workbook.create_sheet(sheet_name_)
        worksheet = workbook[sheet_name_]
        col = 'A'

        for key in df_.keys():
            worksheet[f'{col}1'] = key

            row = 2
            for val in df_[key]:
                if isinstance(val, matplotlib.figure.Figure):
                    val.savefig("temp.png")
                    img = openpyxl.drawing.image.Image("temp.png")
                    # img.width = 500
                    # img.height = 500
                    worksheet.add_image(img, f'{col}{row}')
                else:
                    worksheet[f'{col}{row}'] = val
                row += 1
            
            col = chr(ord(col) + 1)

    workbook.save(filename=filename)


def excelImageWriter(img, mode='path', 
    filename='temp.xlsx', sheet_name='sheet 1', 
    dispose=False, cell='A1'):

    """
        Writes an image in xlsx file
        Args:
            img : image file
            mode :
                'path' => image path
                'fig' => matplotlib figure object
            filename : filename of xlsx file
            dispose : if True, delete the image file after writing it to excel
            cell : which cell to write image
    """

    if mode == 'fig':
        img.savefig('temp.png', dpi=img.dpi)
        img = 'temp.png'

    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.get_worksheet_by_name(sheet_name)
    # print(f"worksheet : {worksheet}")
    worksheet.write_row(0, 0, 'xyz')
    # worksheet.insert_image(cell, img)
    # worksheet = workbook.add_worksheet()
    # worksheet.insert_image(cell, img)
    workbook.close()

    if dispose:
        os.remove(img)