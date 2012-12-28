'''
Created on Dec 27, 2012

@author: dstrauss
'''

import multiprocessing
from multiprocessing.sharedctypes import RawArray
import numpy as np
import time 

class rndk(object):
    glb = []
    def __init__(self,m,n):
        self.glb = np.random.randn(m,n)
#        self.b=np.random.randn(m,n)
#        self.k = k
    def __call__(self,a):
        return np.sum(self.glb[:,a])

def fext(a):
    # print a
    j = 5
    return np.sum(a[0][:j]+a[1][:j])

def fsng(a):
    # print a
    j = 5
    return np.sum(a)

def main():
        
    m = 100000
    n = 80
    
    amp = rndk(m,n)
    
    print 'built objects '


    p = multiprocessing.Pool(processes=4)
    tmr = time.time()
    result = p.map_async(amp, xrange(n))   
     
    r = result.get()
    print 'multi proc time ' + repr(time.time()-tmr)

    p.close()
    
    
    
    k = list()
    
    tmr = time.time()
    for ab in amp.glb.T:
        k.append(np.sum(ab))
    print 'single proc time ' + repr(time.time() - tmr)
    
    print r[:3]
    print k[:3] 
    print 'are they the same? ' + repr(r == k)


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

if __name__=='__main__':
    main()