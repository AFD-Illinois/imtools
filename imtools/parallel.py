# Tools for running embarrassingly parallel operations with multiple processes

import multiprocessing

# Hack for passing lambdas to a parallel function:
# Initialize with a dangling function pointer and fill it at call time
# Credit https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
_func = None

def worker_init(func):
    global _func
    _func = func

def worker(x):
    return _func(x)

def map_parallel(function, input_list, nprocs=None):
    """Run a function in parallel and return a list of all the results. Best for whole-image reductions.
    Takes lambdas thanks to some happy hacking
    """
    with multiprocessing.Pool(nprocs, initializer=worker_init, initargs=(function,)) as p:
        return p.map(worker, input_list)

def iter_parallel(function, merge_function, input_list, output, nprocs=None, initializer=None, initargs=()):
    """Run a function in parallel with Python's multiprocessing
    'function' must not be a lambda, must take only an element of input_list.
    'merge_function' must take the list element number, the return of 'function', and 'output',
    which it must update as an accumulator.  Note merge_function cannot be a lambda as it has a side effect,
    but can be defined locally.
    """
    if initializer is not None:
        pool = multiprocessing.Pool(nprocs, initializer=initializer, initargs=initargs)
    else:
        pool = multiprocessing.Pool(nprocs)

    try:
        # Map the above function to the dump numbers, returning an iterator of 'out' dicts to be merged one at a time
        # This avoids keeping the (very large) full pre-average list in memory
        out_iter = pool.imap(function, input_list)
        for n, result in enumerate(out_iter):
            merge_function(n, result, output)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

def set_mkl_threads(n_mkl):
    try:
        import ctypes
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
        mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads
        mkl_set_num_threads(n_mkl)
        print("Using {} MKL threads".format(mkl_get_max_threads()))
    except Exception as e:
        print(e)