# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:58:39 2021

@author: didie
"""
import multiprocessing as mp
import numpy as np
# Redefine, with only 1 mandatory argument.
def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

def main():
    np.random.RandomState(100)
    arr = np.random.randint(0, 10, size=[200000, 5])
    data = arr.tolist()
    data[:5]
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    
    # Step 2: `pool.apply` the `howmany_within_range()`
    results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
    
    # Step 3: Don't forget to close
    pool.close()    
    
    print(results[:10])
    #> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
    
if __name__ == '__main__':
    main()