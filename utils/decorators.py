import time

def timer(function):
    '''
        Calculates time to execute a function
    '''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = function(*args, **kwargs)
        print("exec time : {} seconds".format(time.time()-start_time))
        return res
    
    return wrapper
