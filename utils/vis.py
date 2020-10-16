import matplotlib.pyplot as plt
import pandas as pd


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
