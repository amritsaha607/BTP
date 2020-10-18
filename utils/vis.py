import matplotlib.pyplot as plt
import pandas as pd

from dataGeneration.utils import getArea


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
    size='auto', debug=True):

    n_rows = 1 if overlay else len(figs)
    n_cols = len(figs) if overlay else len(r['r2'])

    if size=='auto':
        if overlay:
            size = (n_cols*5, 4)
        else:
            size = (n_cols*5, n_rows*4)
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=size)

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
            # label_ = "r1: {}nm, r2: {}nm".format(int(r1*1e9), int(r2*1e9))

            if figs[row]=='sca':
                ax_.plot(lambd*1e9, area_sca, label=label_)
            elif figs[row]=='abs':
                ax_.plot(lambd*1e9, area_abs, label=label_)
            elif figs[row]=='tot':
                ax_.plot(lambd*1e9, area_abs+area_sca, label=label_)
            else:
                raise ValueError("Unknown figure format '{}' found".format(figs[row]))

            ax_.set_xlabel("Wavelength (nm)")
            ax_.set_ylabel("{} Area (m2)".format(figs[row]))
            ax_.legend()

            if overlay:
                ax_.set_title(figs[row])

    plt.show()