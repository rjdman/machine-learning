import multiprocessing
from functools import partial
from tqdm import tqdm_notebook

def batch_multi(parallel_list, parallel_function, batch_size=500, n_processes=multiprocessing.cpu_count()*2, results=True, **kwargs):
    """
    Function to run upload_from_server_to_s3 in parallel
    :param n_processes: number of parallel processes to spawn
    :param batch_size: chunk size of master list
    :param parallel_function: function to parallelise
    :param parallel_list: master list to be batched
    """
    # Create a function called "chunks" with two arguments, l and n:
    def chunks(l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]
    
    batch_list = list(chunks(parallel_list, batch_size))
    p = multiprocessing.Pool(n_processes)
    partial_func = partial(
        parallel_function,
        **kwargs
        )
    result = list(tqdm_notebook(p.imap(partial_func, batch_list), total=len(batch_list)))
    p.close()
    p.join()
    if results:
        return result