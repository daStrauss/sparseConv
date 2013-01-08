'''
Created on Jan 4, 2013

@author: dstrauss
'''
import numpy as np
class foo(object):
    def __init__(self,x):
        self.x = x
        
    def go(self,y):
        return self.x*y
    
    
class bar(foo):
    def go(self,y,z):
        return self.go(y) + z
    

def grr(A):
    z = list()
    for a in A:
        x = np.random.rand()
        z.append(x)
    return z