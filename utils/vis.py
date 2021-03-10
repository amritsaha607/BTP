import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb

from dataGeneration.utils import getArea


def plotArr(arr, debug=True,
            size_=(8, 8), scatter=False, continuous=False):
    """
        Plot a list or 1d array
    """

    if isinstance(arr, list):
        n = len(arr)
    else:
        n = arr.shape[0]
    
    if not debug:
        fig = plt.figure(figsize=size_)
    
    x_range = np.arange(0, n, 1.0)
    if scatter:
        plt.scatter(x_range, arr)
    if continuous:
        plt.plot(x_range, arr)
    
    if not debug:
        plt.close()
        return fig
    
    plt.show()
    

def plotArrMulti(arr, labels='temp_label', debug=True,
            size_=(8, 8), scatter=False, continuous=False,
            ret_mode='fig'):
    """
        Multiple plot of arrays in one plot
    """

    if not scatter and not continuous:
        raise AssertionError("Please set at least one of scatter and continuous parameter")

    if isinstance(arr, list):
        n = len(arr)
    else:
        n = arr.shape[0]

    if isinstance(labels, str):
        labels = [labels]*n

    if not debug:
        fig = plt.figure(figsize=size_)

    for arr_, label in zip(arr, labels):
        n = len(arr_) if isinstance(arr_, list) else arr_.shape[0]
        x_range = np.arange(0, n, 1.0)
        if scatter:
            plt.scatter(x_range, arr_, label=label)
        if continuous:
            plt.plot(x_range, arr_, label=label)

    plt.legend()
    if not debug:
        if ret_mode == 'wandb_image':
            fig = wandb.Image(fig)
        plt.close()
        return fig

    plt.show()
    



def plotCSV(f_name, debug=True, 
            size_=(8, 8), scatter=False, continuous=True,
            cols=['n', 'k', 'er', 'ei']):
    '''
        Gets n, k, er, ei from CSV with varying wavelength and plots them w.r.t. wavelength
    '''
    content = pd.read_csv(f_name)
    
    if not debug:
        fig = plt.plot(figsize=size_)
    
    if scatter:
        for col in cols:
            plt.scatter(content['wl'], content[col], label=col)
    if continuous:
        for col in cols:
            plt.plot(content['wl'], content[col], label=col)

    plt.legend()

    if not debug:
        # plt.close()
        return fig
    
    plt.show()


def plotArea(r, eps, lambd,
    figs=['sca', 'abs', 'tot'], overlay=False, label_mode=['r1', 'r2'],
    size='auto', debug=True, title='auto'):

    """
        Function to plot area w.r.t. varying wavelength for different parameters
        Args:
            r : (dict) radius data
            eps : (dict) epsilon data
            lamd : (float array) varying wavelength data
            figs : (list) figures to show
                    'sca': scattering area
                    'abs': absorption area
                    'tot': total area
            overlay : (bool) To overlap the plots for different parameters or not
            label_mode : (list) label with r1 or r2 or both values
            size : figure size ('auto' for auto adjustments)
            debug : set True while plotting in notebook
                    set False if you wanna get the figure object
    """

    n_rows = 1 if overlay else len(figs)
    n_cols = len(figs) if overlay else len(r['r2'])

    # Generate size for 'auto' size mode
    if size=='auto':
        if overlay:
            size = (n_cols*5, 4)
        else:
            size = (n_cols*5, n_rows*4)
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=size)

    # Make plots
    for col, (r1, r2) in enumerate(zip(r['r1'], r['r2'])):
        r_cp = r.copy()
        r_cp['r1'] = r1
        r_cp['r2'] = r2
        area_sca, area_abs = getArea(r_cp, eps, lambd)

        for row in range(len(figs)):
            if overlay:
                ax_ = ax[row] if n_cols>1 else ax
            else:
                ax_ = ax[row][col] if n_rows>1 else ax[col]
            
            label_ = []
            if 'r1' in label_mode:
                label_.append("r1: {}nm".format(int(r1*1e9)))
            if 'r2' in label_mode:
                label_.append("r2: {}nm".format(int(r2*1e9)))
            label_ = ', '.join(label_)

            if figs[row]=='sca':
                ax_.plot(lambd*1e9, area_sca, label=label_)
            elif figs[row]=='abs':
                ax_.plot(lambd*1e9, area_abs, label=label_)
            elif figs[row]=='tot':
                ax_.plot(lambd*1e9, area_abs+area_sca, label=label_)
            else:
                raise ValueError("Unknown figure format '{}' found".format(figs[row]))

            ax_.set_xlabel("Wavelength (nm)")
            ax_.set_ylabel("Cross Section (m2)")
            ax_.legend()

            title_ = title
            if overlay and title=='auto':
                title_ = 'Total Cross Section'
                if figs[row]=='abs':
                    title_ = 'Absorption Cross Section'
                elif figs[row]=='sca':
                    title_ = 'Scattering Cross Section'

            ax_.set_title(title_)

    if debug:
        plt.show()
    
    else:
        plt.close()
        return fig