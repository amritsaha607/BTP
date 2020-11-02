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