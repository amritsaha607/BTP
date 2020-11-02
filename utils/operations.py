import numpy as np
import torch


def makeList(x, n):
    '''
        Makes a variable list
        Args:
            x : variable to be made a list
            n : length of the list
        Returns:
            List object
    '''

    if isinstance(x, (int, float)):
        x = [x]*n

    elif isinstance(x, list):
        if not len(x)==n:
            raise ValueError("Cannot convert a list of size {} to list of size {}".format(len(x), n))

    return x


def listAdd(lists):
    """
        Add corresponding values of list of lists
    """
    res = np.array([0]*len(lists[0]))
    lists = np.array(lists)
    for elem in lists:
        res += elem
    return res


def dictMultiply(d1, d2):
    """
        in case any key is not found in d2, the d1 value will be kept in res
    """
    keys = d1.keys()
    res = {}
    for key in keys:
        if not key in d2.keys():
            res[key] = d1[key]
        elif isinstance(d1[key], dict):
            res[key] = dictMultiply(d1[key], d2[key])
        elif isinstance(d1[key], list):
            res[key] = np.array(d1[key])*np.array(d2[key])
        else:
            res[key] = d1[key]*d2[key]
    return res


def dictAdd(ld, weights=None):
    """
        Add corresponding keys of list of dicts
        no matter how nested the dict is
        Args:
            ld : List of dicts
            weights : for weighted sum
        Returns:
            res : Final dict with added values
    """

    n = len(ld)
    keys = ld[0].keys()
    res = {}

    if weights is not None:
        for i in range(n):
            ld[i] = dictMultiply(ld[i], weights[i])

    for key in keys:
        if isinstance(ld[0][key], dict):
            res[key] = dictAdd([ld[i][key] for i in range(n)])
        elif isinstance(ld[0][key], list):
            res[key] = listAdd([ld[i][key] for i in range(n)])
        else:
            res[key] = sum([ld[i][key] for i in range(n)])

    return res
