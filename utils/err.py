import numpy as np

def getErr(a, b, lump=False):
    '''
        Calculates error between two data
        Args:
            a : data 1 (can be int, float, list, dict)
            b : data 2 (do)
            lump : In case a & b are list/dict
                lump='all' will give avg error of all the corresponding elements
                lump='max' will give max error of all the corresponding elements
                lump=False will restore the original data structure and return error of individual element
    '''
    # Int / Float
    if isinstance(a, int) or isinstance(a, float):
        return abs(a-b)/b

    # List
    elif isinstance(a, list):
        if lump=='all':
            return abs(sum(a)-sum(b))/sum(b)
        else:
            d = np.array(a)-np.array(b)
            d /= np.array(b)
            if lump=='max':
                return np.max(d)
            else:
                return d
    
    # Dict (To be implemented)