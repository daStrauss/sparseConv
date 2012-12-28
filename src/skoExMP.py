'''
Created on Dec 27, 2012

@author: dstrauss
'''

from multiprocessing import Pool, Array, Process

def count_it( key ):
    count = 0
    for c in toShare:
        if c == key:
            count += 1
    return count

if __name__ == '__main__':
    # allocate shared array - want lock=False in this case since we 
    # aren't writing to it and want to allow multiple processes to access
    # at the same time - I think with lock=True there would be little or 
    # no speedup
    maxLength = 50
    toShare = Array('c', maxLength, lock=False)
    
    # fork
    pool = Pool()
    
    # can set data after fork
    testData = "abcabcs bsdfsdf gdfg dffdgdfg sdfsdfsd sdfdsfsdf"
    if len(testData) > maxLength:
        raise ValueError, "Shared array too small to hold data"
    toShare[:len(testData)] = testData

    print pool.map( count_it, ["a", "b", "s", "d"] )