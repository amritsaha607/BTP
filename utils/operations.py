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